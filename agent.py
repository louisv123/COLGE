import numpy as np
import random
import time
import logging
import models
from utils.config import load_model_config

import torch.nn.functional as F
import torch



# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

"""
Contains the definition of the agent that will run in an
environment.
"""




class DQAgent:


    def __init__(self,graph,model):

        self.graphs = graph
        self.embed_dim = 64
        self.model_name = model

        self.k = 20
        self.alpha = 0.1
        self.gamma = 0.99
        self.lambd = 0.
        self.n_step=5

        self.epsilon=1
        self.epsilon_min=0.05
        self.discount_factor =0.99995# 0.9998
        # self.games = 0
        self.t=1
        self.memory = []
        self.memory_n=[]

        if self.model_name == 'S2V_QN_1':

            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_1(**args_init)

        elif self.model_name == 'LINE_QN':

            args_init = load_model_config()[self.model_name]
            self.model = models.LINE_QN(**args_init)

        elif self.model_name == 'W2V_QN':

            args_init = load_model_config()[self.model_name]
            self.model = models.W2V_QN(G=self.graphs[self.games], **args_init)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.e-5)
        self.T = 5




    """
    p : embedding dimension
       
    """

    def reset(self, g):

        self.t=1
        #self.memory=[]
        #self.memory_n=[]
        self.games = g
        #self.epsilon=1

        if (len(self.memory) != 0) and (len(self.memory) % 300000 == 0):
            self.memory = random.sample(self.memory,120000)

        if (len(self.memory_n) != 0) and (len(self.memory_n) % 300000 == 0):
            self.memory_n =random.sample(self.memory_n,120000)

        self.minibatch_length = 128

        self.nodes = self.graphs[self.games].nodes()
        self.adj = self.graphs[self.games].adj()
        self.adj = self.adj.todense()
        self.adj = torch.from_numpy(np.expand_dims(self.adj.astype(int), axis=0))
        self.adj = self.adj.type(torch.FloatTensor)

        self.last_action = 0
        self.last_observation = torch.zeros(1, self.nodes, 1, dtype=torch.float)
        self.last_reward = -0.1




    def act(self, observation):


        if self.epsilon > np.random.rand():
            return np.random.choice(np.where(observation.numpy()[0,:,0] == 0)[0])
        else:
            q_a = self.model(observation, self.adj)
            q_a=q_a.detach().numpy()
            return np.where((q_a[0, :, 0] == np.max(q_a[0, :, 0][observation.numpy()[0, :, 0] == 0])))[0][0]

    def reward(self, observation, action, reward,done):

        if len(self.memory_n) > self.minibatch_length + self.n_step or self.games > 2:

            (last_observation_tens, action_tens, reward_tens, observation_tens, adj_tens) = self.get_sample()
            target = reward_tens + self.gamma *torch.max(self.model(observation_tens, adj_tens) + observation_tens * (-1e5), dim=1)[0]
            target_f = self.model(last_observation_tens, adj_tens)
            target_p = target_f.clone()
            target_f[range(self.minibatch_length),action_tens,:] = target
            loss=self.criterion(target_p, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(self.t, loss)

            if self.epsilon > self.epsilon_min:
               self.epsilon *= self.discount_factor

        self.remember(self.last_observation, action, self.last_reward, observation.clone())

        if self.t > self.n_step:
            self.remember_n(done)
        self.t += 1
        self.last_action = action
        self.last_observation = observation.clone()
        self.last_reward = reward

    def get_sample(self):

        minibatch = random.sample(self.memory_n, self.minibatch_length - 1)
        minibatch.append(self.memory_n[-1])
        last_observation_tens = minibatch[0][0]
        action_tens = torch.Tensor([minibatch[0][1]]).type(torch.LongTensor)
        reward_tens = torch.Tensor([[minibatch[0][2]]])
        observation_tens = minibatch[0][3]
        adj_tens = self.graphs[minibatch[0][4]].adj().todense()
        adj_tens = torch.from_numpy(np.expand_dims(adj_tens.astype(int), axis=0)).type(torch.FloatTensor)

        for last_observation_, action_, reward_, observation_, games_ in minibatch[-self.minibatch_length + 1:]:
            last_observation_tens=torch.cat((last_observation_tens,last_observation_))
            action_tens = torch.cat((action_tens, torch.Tensor([action_]).type(torch.LongTensor)))
            reward_tens = torch.cat((reward_tens, torch.Tensor([[reward_]])))
            observation_tens = torch.cat((observation_tens, observation_))
            adj_ = self.graphs[self.games].adj().todense()
            adj = torch.from_numpy(np.expand_dims(adj_.astype(int), axis=0)).type(torch.FloatTensor)
            adj_tens = torch.cat((adj_tens, adj))

        return (last_observation_tens, action_tens, reward_tens, observation_tens, adj_tens)



    def remember(self, last_observation, last_action, last_reward, observation):
        self.memory.append((last_observation, last_action, last_reward, observation, self.games))

    def remember_n(self,done):

        if not done:
            step_init = self.memory[-self.n_step]
            cum_reward=step_init[2]
            for step in range(1,self.n_step):
                cum_reward+=self.memory[-step][2]
            self.memory_n.append((step_init[0], step_init[1], cum_reward, self.memory[-1][-2], self.games))

        else:
            for i in range(1,self.n_step):
                step_init = self.memory[-self.n_step+i]
                cum_reward=step_init[2]
                for step in range(1,self.n_step-i):
                    cum_reward+=self.memory[-step][2]
                self.memory_n.append((step_init[0], step_init[1], cum_reward, self.memory[-1][-2], self.games))


Agent = DQAgent
