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

        #self.graph=graph
        self.nodes=graph.nodes()
        self.adj = graph.adj()
        self.adj=self.adj.todense()
        self.adj=torch.from_numpy(np.expand_dims(self.adj.astype(int),axis=0))
        self.adj = self.adj.type(torch.FloatTensor)
        self.embed_dim = 64

        self.k = 20
        self.alpha = 0.1
        self.gamma = 0.99
        self.lambd = 0.
        self.n_step=5

        self.epsilon=1
        self.epsilon_min=0.05
        self.discount_factor=0.995
        self.games = 0
        self.t=1
        self.memory=[]
        self.memory_n=[]
        self.minibatch_length = 16

        self.last_action = 0
        self.last_observation = torch.zeros(1,self.nodes,1, dtype=torch.float)
        self.last_reward = -1
        self.mu_init = torch.zeros(1,self.nodes,self.embed_dim,dtype=torch.float)

        if model=='S2V_QN':

            args_init = load_model_config()[model]
            self.model =models.S2V_QN(**args_init)

        elif model=='LINE_QN':

            args_init = load_model_config()[model]
            self.model = models.LINE_QN(**args_init)

        elif model=='W2V_QN':

            args_init = load_model_config()[model]
            self.model = models.W2V_QN(G=graph.g,**args_init)


        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.e-5)
        self.T=5


    """
    p : embedding dimension
       
    """


    def reset(self):

        self.t=1
        #self.memory=[]
        #self.memory_n=[]
        self.games += 1
        #self.epsilon=1
        self.last_action = 0
        self.last_observation = torch.zeros(1,self.nodes, 1, dtype=torch.float)
        self.last_reward = 0


    def act(self, observation):


        if self.epsilon > np.random.rand():
            return np.random.choice(np.where(observation.numpy()[0,:,0] == 0)[0])
        else:
            q_a = self.model(observation,self.adj,self.mu_init)
            q_a=q_a.detach().numpy()
            return np.where((q_a[0,:,0]==np.max(q_a[observation.numpy()[:,:,0]==0])))[0][0]

    def reward(self, observation, action, reward,done):



        if self.t > self.minibatch_length+self.n_step or self.games>2:

            (last_observation_tens,action_tens,reward_tens,observation_tens) = self.get_sample()
            target = reward_tens + self.gamma * torch.max(self.model(observation_tens,self.adj,self.mu_init)+observation_tens*(-1e5),dim=1)[0]
            target_f = self.model(last_observation_tens,self.adj,self.mu_init)
            target_p = target_f.clone()
            target_f[range(self.minibatch_length),action_tens,:] = target
            loss=self.criterion(target_p, target_f)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #logging.info('Loss for t = %s is %f' % (self.t, loss))

            if self.epsilon > self.epsilon_min:
               self.epsilon *= self.discount_factor

        self.remember(self.last_observation, self.last_action, self.last_reward, observation.clone())

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

        for last_observation_, action_, reward_, observation_ in minibatch[-self.minibatch_length+1:]:
            last_observation_tens=torch.cat((last_observation_tens,last_observation_))
            action_tens = torch.cat((action_tens, torch.Tensor([action_]).type(torch.LongTensor)))
            reward_tens = torch.cat((reward_tens, torch.Tensor([[reward_]])))
            observation_tens = torch.cat((observation_tens, observation_))

        return (last_observation_tens,action_tens,reward_tens,observation_tens)



    def remember(self, last_observation, last_action, last_reward, observation):
        self.memory.append((last_observation, last_action, last_reward, observation))

    def remember_n(self,done):

        if not done:
            step_init = self.memory[-self.n_step]
            cum_reward=step_init[2]
            for step in range(1,self.n_step):
                cum_reward+=self.memory[-step][2]
            self.memory_n.append((step_init[0],step_init[1],cum_reward,self.memory[-1][-1]))

        else:
            for i in range(1,self.n_step):
                step_init = self.memory[-self.n_step+i]
                cum_reward=step_init[2]
                for step in range(1,self.n_step-i):
                    cum_reward+=self.memory[-step][2]
                self.memory_n.append((step_init[0],step_init[1],cum_reward,self.memory[-1][-1]))


Agent = DQAgent
