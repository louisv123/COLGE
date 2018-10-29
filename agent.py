import numpy as np
import random
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Add, Dot, Concatenate, ReLU
from keras.optimizers import sgd

"""
Contains the definition of the agent that will run in an
environment.
"""


class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change when starting a new
        episode of the same game, but your initial location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        self.state_space = [0,20]
        #self.state_space = [[x_range[0] + x_range[0] * i / (state_bucket[0] - 1) for i in range(state_bucket[0])], [-20 + j * 40 / (state_bucket[1] - 1) for j in range(state_bucket[1])]]

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """

        # run your code

        observation_ = self.state_to_bucket(observation)

        return np.random.choice([-1, 0, 1])

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass



class q_learning_agent_2:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 0.1
        self.state_bucket = (12, 12)

        self.action_space = [-1, 0, 1]

        self.discount_factor = 0.9
        self.learning_rate = 0.002
        self.lambda_ = 0.8
        self.t = 1
        self.last_observation = (-100, 0)
        self.last_last_action = 0
        self.last_action = 0
        x_range = [-150, 0]
        self.state_space = [[x_range[0] + -x_range[0] * i / (self.state_bucket[0] - 1) for i in range(self.state_bucket[0])], [-20 + j * 40 / (self.state_bucket[1] - 1) for j in range(self.state_bucket[1])]]
        """
        self.W = np.random.rand(self.state_bucket[0] * self.state_bucket[1] * len(self.action_space) + 1)
        """
        self.W_ = np.random.rand(len(self.action_space), self.state_bucket[0] * self.state_bucket[1] + 1)

        self.Phi_ = np.ones(self.state_bucket[0] * self.state_bucket[1] + 1)

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change when starting a new
        episode of the same game, but your initial location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        print(self.t)
        """
        if self.t < 500:
            self.W_ = np.random.rand(len(self.action_space), self.state_bucket[0] * self.state_bucket[1] + 1)
        """

        self.t = 1
        self.time_start = time.clock()

        """
        print(self.Phi_function(self.last_observation, self.last_action))
        print(self.W_)
        plt.subplot(3, 1, 1)
        plt.contour(self.state_space[0], self.state_space[1], self.q_matrix(-1))
        plt.colorbar()
        plt.subplot(3, 1, 2)
        plt.contour(self.state_space[0], self.state_space[1], self.q_matrix(0))
        plt.colorbar()
        plt.subplot(3, 1, 3)
        plt.contour(self.state_space[0], self.state_space[1], self.q_matrix(1))
        plt.colorbar()
        plt.show()
        """

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)



        observation_ = self.state_to_bucket(observation)

        index_observation = ()
        for dim in range(len(self.state_bucket)):
            index_observation += (self.state_space[dim].index(observation_[dim]),)

        time_elapsed = (time.clock() - self.time_start)
        print(time_elapsed)
        self.time_start = time.clock()
        """

        if random.random() < self.epsilon:
            return np.random.choice([-1, 0, 1])
        else:
            q = self.Q_function(observation, None)
            return self.action_space[np.random.choice(np.flatnonzero(q == q.max()))]

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        action_ = action
        if action == None:
            action_ = 0

        """
        observation_ = self.state_to_bucket(observation)



        index_observation = ()
        for dim in range(len(self.state_bucket)):
            index_observation += (self.state_space[dim].index(observation_[dim]),)
        """
        index_action = (self.action_space.index(action_),)
        index_last_action = (self.action_space.index(self.last_action),)
        index_last_last_action = (self.action_space.index(self.last_last_action),)

        if self.t == 1:
            self.last_action = np.random.choice(self.action_space)
            self.e_1 = 0
            self.e_2 = 0
        elif self.t == 2:

            best_q = np.amax(self.Q_function(observation, None))

            target = self.last_reward + self.discount_factor * (best_q)
            self.e_2 = self.discount_factor * self.lambda_ * self.e_2 + self.Phi_function(self.last_observation)
            delta_2 = target - np.dot(self.W_[index_last_action, :], self.Phi_)
            self.W_[index_last_action, :] += self.learning_rate * delta_2 * self.e_2

        else:
            """
            best_q = np.amax(self.q_table[index_observation])
            self.q_table[self.last_observation + self.last_action] += self.learning_rate * (reward + self.discount_factor * (best_q) - self.q_table[self.last_observation + self.last_action])
            """

            best_q = np.amax(self.Q_function(observation, None))

            target_1 = self.last_last_reward + self.discount_factor * self.last_reward + self.discount_factor * (best_q)
            target_2 = self.last_reward + self.discount_factor * (best_q)
            self.e_1 = self.discount_factor * self.lambda_ * self.e_1 + self.Phi_function(self.last_last_observation)
            delta_1 = target_1 - np.dot(self.W_[index_last_last_action, :], self.Phi_)
            self.e_2 = self.discount_factor * self.lambda_ * self.e_2 + self.Phi_function(self.last_observation)

            delta_2 = target_2 - np.dot(self.W_[index_last_action, :], self.Phi_)
            self.W_[index_last_last_action, :] += self.learning_rate * delta_1 * self.e_1
            self.W_[index_last_action, :] += self.learning_rate * delta_2 * self.e_2

        self.t += 1
        self.last_action = action_
        self.last_observation = observation
        self.last_reward = reward
        self.last_last_reward = self.last_reward
        self.last_last_action = self.last_action
        self.last_last_observation = self.last_observation
    """
    def state_to_bucket(self, state):
        observation_ = ()
        for dim in range(len(self.state_bucket)):
            distance = list()
            for state_ in self.state_space[dim]:
                distance.append(np.abs(state_ - state[dim]))
            observation_ += (self.state_space[dim][np.argmin(distance)],)

        return observation_


    def set_q_table(self):
        seq_dim = ()
        for dim in range(len(self.state_space)):
            seq_dim = seq_dim + (len(self.state_space[dim]),)
        seq_dim = seq_dim + (len(self.action_space),)
        self.q_table = np.zeros(seq_dim)

    """

    def Phi_function(self, state):
        """

        action_ = action
        if action == None:
            action_ = 0



        index_action = self.action_space.index(action_)



        self.Phi = np.ones(self.state_bucket[0] * self.state_bucket[1] * len(self.action_space) + 1)
        self.coef_act = np.zeros(self.state_bucket[0] * self.state_bucket[1] * len(self.action_space) + 1)

        self.coef_act[-1] = 1

        for i in range(self.state_bucket[0]):
            for j in range(self.state_bucket[1]):
                for a in range(len(self.action_space)):
                    self.Phi[i + self.state_bucket[0] * j + self.state_bucket[0] * self.state_bucket[1] * a] = (np.exp(-((state[0] - self.state_space[0][i]) / (self.state_bucket[0]))**2) * np.exp(-((state[1] - self.state_space[1][j]) / (self.state_bucket[1]))**2))

        self.coef_act[index_action * self.state_bucket[0] * self.state_bucket[1]:(index_action + 1) * self.state_bucket[0] * self.state_bucket[1]] = 1

        self.Phi = np.multiply(self.coef_act, self.Phi)

        return self.Phi

        """

        for i in range(self.state_bucket[0]):
            for j in range(self.state_bucket[1]):
                self.Phi_[i + self.state_bucket[0] * j] = (np.exp(-((state[0] - self.state_space[0][i]) / (self.state_bucket[0]))**2) * np.exp(-((state[1] - self.state_space[1][j]) / (self.state_bucket[1]))**2))

        return self.Phi_

        """

        self.Phi_ = {}
        for i in range(self.state_bucket[0]):
            for j in range(self.state_bucket[1]):
                self.Phi_[(i, j)] = (np.exp(-((state[0] - self.state_space[0][i]) / (self.state_bucket[0]))**2) * np.exp(-((state[1] - self.state_space[1][j]) / (self.state_bucket[1]))**2))
        self.Phi_[(40, 39)] = 1
        return np.array(self.Phi_.values())


        Ph[state,action]=[0,0,0,ph_1(state,action),ph_2(state,action),ph_(state,action),0,0,0,1]
        Ph_size=1+nstate*naction

        W=[wa=1_1,...wa=1_ns,wa=2_1,...,wa=2_ns,wa=3_1,...,wa=3_ns,w0]
        """

    def Q_function(self, state, action):
        """
        return np.dot(self.W, self.Phi_function(state, action))

        """

        return np.dot(self.W_, self.Phi_function(state))
    """
        else:

            index_action = self.action_space.index(action)
            return np.dot(self.W_[index_action, :], self.Phi_function(state))



    def q_matrix(self, action):

        return [[self.Q_function((self.state_space[0][i], self.state_space[1][j]), action) for i in range(self.state_bucket[0])] for j in range(self.state_bucket[1])]
    """


class q_learning_agent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 0.1
        self.state_bucket = (12, 12)

        self.action_space = [-1, 0, 1]

        self.discount_factor = 0.9
        self.learning_rate = 0.002
        self.lambda_ = 0.8
        self.t = 1
        self.last_observation = (-100, 0)
        self.last_action = 0

        x_range = [-150, 0]
        self.state_space = [[x_range[0] + -x_range[0] * i / (self.state_bucket[0] - 1) for i in range(self.state_bucket[0])], [-20 + j * 40 / (self.state_bucket[1] - 1) for j in range(self.state_bucket[1])]]

        self.W_ = np.random.rand(len(self.action_space), self.state_bucket[0] * self.state_bucket[1] + 1)

        self.Phi_ = np.ones(self.state_bucket[0] * self.state_bucket[1] + 1)

    def reset(self, x_range):

        print(self.t)
        self.t = 1

        """
        print(self.Phi_function(self.last_observation, self.last_action))
        print(self.W_)
        plt.subplot(3, 1, 1)
        plt.contour(self.state_space[0], self.state_space[1], self.q_matrix(-1))
        plt.colorbar()
        plt.subplot(3, 1, 2)
        plt.contour(self.state_space[0], self.state_space[1], self.q_matrix(0))
        plt.colorbar()
        plt.subplot(3, 1, 3)
        plt.contour(self.state_space[0], self.state_space[1], self.q_matrix(1))
        plt.colorbar()
        plt.show()
        """

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if random.random() < self.epsilon:
            return np.random.choice([-1, 0, 1])
        else:
            q = self.Q_function(observation, None)
            return self.action_space[np.random.choice(np.flatnonzero(q == q.max()))]

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        action_ = action
        if action == None:
            action_ = 0

        index_action = (self.action_space.index(action_),)
        index_last_action = (self.action_space.index(self.last_action),)

        if self.t == 1:
            self.last_action = np.random.choice(self.action_space)
            self.e = 0

        else:
            """
            best_q = np.amax(self.q_table[index_observation])
            self.q_table[self.last_observation + self.last_action] += self.learning_rate * (reward + self.discount_factor * (best_q) - self.q_table[self.last_observation + self.last_action])
            """

            best_q = np.amax(self.Q_function(observation, None))

            target = self.last_reward + self.discount_factor * (best_q)
            self.e = self.discount_factor * self.lambda_ * self.e + self.Phi_function(self.last_observation)
            delta = target - np.dot(self.W_[index_last_action, :], self.Phi_)
            self.W_[index_last_action, :] += self.learning_rate * delta * self.e

        self.t += 1
        self.last_action = action_
        self.last_observation = observation
        self.last_reward = reward

    def Phi_function(self, state):

        for i in range(self.state_bucket[0]):
            for j in range(self.state_bucket[1]):
                self.Phi_[i + self.state_bucket[0] * j] = (np.exp(-((state[0] - self.state_space[0][i]) / (self.state_bucket[0]))**2) * np.exp(-((state[1] - self.state_space[1][j]) / (self.state_bucket[1]))**2))

        return self.Phi_

    def Q_function(self, state, action):

        return np.dot(self.W_, self.Phi_function(state))

class TDQAgent:


    def __init__(self,graph):

        #self.graph=graph
        self.nodes=graph.nodes()
        self.adj = graph.adj()
        self.adj=self.adj.todense()
        self.p = 50

        self.k = 20
        self.alpha = 0.1
        self.gamma = 0.99
        self.lambd = 0.99

        self.epsilon=0.1
        self.games = 0
        self.t=1
        self.memory=[]
        self.minibacth_lentgh = 8

        self.last_action = 0
        self.last_observation = np.zeros(self.nodes)
        self.last_reward = -1
        self.mu_init = np.zeros((self.p, self.nodes))

        self.model = self.build_model(T=4)

        #### Build model #######
    def build_model(self,T):

        xv = Input(batch_shape=(1, self.nodes, 1))
        mu_init = Input(batch_shape=(1, self.nodes, p))
        adj = Input(batch_shape=(1, self.nodes, self.nodes))

        for t in range(T):
            if t == 0:
                mu_1 = Dense(self.p, input_shape=(1,self. nodes, 1))(xv)
                mu_2 = Dense(self.p, input_shape=(1, self.nodes, self.p))(Dot(axes=1)([adj, mu_init]))
                mu = ReLU()(Add()([mu_1, mu_2]))
            else:
                mu_1 = Dense(self.p, input_dim=self.nodes)(xv)
                mu_2 = Dense(self.p, input_dim=self.nodes)(Dot(axes=1)([adj, mu]))
                mu = ReLU()(Add()([mu_1, mu_2]))

        q_1 = Dense(self.p, input_dim=self.p)(Dot(axes=1)([adj, mu]))
        q_2 = Dense(self.p, input_dim=self.p)(mu)
        q_ = Concatenate(axis=0)([q_1, q_2])
        q = Dense(1, activation="relu")(q_)

        model = Model(inputs=[xv, mu_init, adj], outputs=q)
        model.compile(optimizer='rmsprop',
                      loss='mse')

    """
    p : embedding dimension
    theta1: S2V parameter
    theta2: S2V parameter
    theta5: Q parameter
    theta6: Q parameter
    theta7: Q parameter
    
       
    """


    def reset(self):

        self.xv=np.zeros(self.nodes)

        self.t=1
        self.memory=[]
        self.games += 1


    def act(self, observation):


        if self.games < 190 and self.epsilon > np.random.rand():
            return np.random.choice(np.flatnonzero(self.xv_bar == 1))
        else:
            q_a = self.model.predict(x=[observation,self.mu_init,self.adj])

        return np.argmax(q_a)

    def reward(self, observation, action, reward):

        if self.t > self.minibatch_length:
            minibatch = random.sample(self.memory, self.minibatch_length - 1)
            minibatch.append(self.memory[-1])
            for self.last_observation, action_, reward_, observation_ in minibatch:
                target = (reward_ + self.gamma * np.amax(self.model.predict(x=[observation_,self.mu_init,self.adj])[-1]))
                target_f = self.model.predict(x=[self.last_observation,self.mu_init,self.adj])
                target_f[-1][action_] = target
                self.model.fit(self.last_observation, target_f, epochs=1, verbose=0)
            # if self.epsilon > self.epsilon_min:
            #   self.epsilon *= self.discount_factor

        self.remember(self.last_observation, self.last_action, self.last_reward, observation)
        self.t += 1
        self.last_action = action
        self.last_observation = observation
        self.last_reward = reward


    def remember(self, last_observation, last_action, last_reward, observation):
        self.memory.append((last_observation, last_action, last_reward, observation))



class PolicyAgent:
    def __init__(self):
        self.p = 5
        self.k = 3
        self.w_mu = np.zeros((self.p + 1, self.k + 1))
        self.w_value = np.zeros((self.p + 1, self.k + 1))
        self.gamma = 0.99
        self.lmbda = 0.99
        self.alpha = 0.01
        self.alpha_v = 0.02
        self.games = 0

    def reset(self, x_range):
        x_min = float(x_range[0])
        self.x_anchors = np.array([[x_min + i - x_min / self.p] for i in range(self.p + 1)])
        self.v_anchors = np.array([[-20.0 + i * 40.0 / self.k for i in range(self.k + 1)]])
        self.xrate = x_min / self.p
        self.vrate = 40 / self.k
        self.e_mu = np.zeros((self.p + 1, self.k + 1))
        self.e_value = np.zeros((self.p + 1, self.k + 1))
        self.last = None
        self.games += 1
        if self.games < 180:
            self.log_sigma = 2.0 * (0.99 ** self.games)
        else:
            self.log_sigma = -1

    def grid(self, obs):
        (x, v) = obs
        x_vals = np.exp(-((self.x_anchors - x) / self.xrate) ** 2)
        v_vals = np.exp(-((self.v_anchors - v) / self.vrate) ** 2)
        return x_vals * v_vals

    def act(self, obs):
        s = self.grid(obs)

        if self.last is not None and self.games < 180:
            # only learn _before_ exploitation
            (old_o, old_a, old_r) = self.last
            old_s = self.grid(old_o)
            old_mu = np.sum(self.w_mu * old_s)
            old_sigma = np.exp(self.log_sigma)
            old_value = np.sum(self.w_value * old_s)
            new_value = np.sum(self.w_value * s)
            # update e
            grad_w_mu_log_pi = (old_a - old_mu) / (old_sigma ** 2) * old_s
            self.e_mu = self.lmbda * self.e_mu + grad_w_mu_log_pi
            # update policy
            update = (old_r + self.gamma * new_value - old_value)
            self.w_mu += self.alpha * update * self.e_mu
            self.w_mu = np.clip(self.w_mu, -4, 4)
            # update value
            self.e_value = self.lmbda * self.e_value + old_s
            self.w_value += self.alpha_v * update * self.e_value

        mu = np.sum(self.w_mu * s)
        sigma = np.exp(self.log_sigma)
        return np.random.normal(mu, sigma)

    def reward(self, obs, act, rew):
        self.last = (obs, act, rew)

class s2v_q_learning_agent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 0.1
        self.state_bucket = (12, 12)

        self.action_space = [-1, 0, 1]

        self.discount_factor = 0.9
        self.learning_rate = 0.002
        self.lambda_ = 0.8
        self.t = 1
        self.last_observation = (-100, 0)
        self.last_action = 0

        x_range = [-150, 0]
        self.state_space = [[x_range[0] + -x_range[0] * i / (self.state_bucket[0] - 1) for i in range(self.state_bucket[0])], [-20 + j * 40 / (self.state_bucket[1] - 1) for j in range(self.state_bucket[1])]]

        self.W_ = np.random.rand(len(self.action_space), self.state_bucket[0] * self.state_bucket[1] + 1)

        self.Phi_ = np.ones(self.state_bucket[0] * self.state_bucket[1] + 1)

    def reset(self, x_range):

        print(self.t)
        self.t = 1

        """
        print(self.Phi_function(self.last_observation, self.last_action))
        print(self.W_)
        plt.subplot(3, 1, 1)
        plt.contour(self.state_space[0], self.state_space[1], self.q_matrix(-1))
        plt.colorbar()
        plt.subplot(3, 1, 2)
        plt.contour(self.state_space[0], self.state_space[1], self.q_matrix(0))
        plt.colorbar()
        plt.subplot(3, 1, 3)
        plt.contour(self.state_space[0], self.state_space[1], self.q_matrix(1))
        plt.colorbar()
        plt.show()
        """

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if random.random() < self.epsilon:
            return np.random.choice([-1, 0, 1])
        else:
            q = self.Q_function(observation, None)
            return self.action_space[np.random.choice(np.flatnonzero(q == q.max()))]

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        action_ = action
        if action == None:
            action_ = 0

        index_action = (self.action_space.index(action_),)
        index_last_action = (self.action_space.index(self.last_action),)

        if self.t == 1:
            self.last_action = np.random.choice(self.action_space)
            self.e = 0

        else:
            """
            best_q = np.amax(self.q_table[index_observation])
            self.q_table[self.last_observation + self.last_action] += self.learning_rate * (reward + self.discount_factor * (best_q) - self.q_table[self.last_observation + self.last_action])
            """

            best_q = np.amax(self.Q_function(observation, None))

            target = self.last_reward + self.discount_factor * (best_q)
            self.e = self.discount_factor * self.lambda_ * self.e + self.Phi_function(self.last_observation)
            delta = target - np.dot(self.W_[index_last_action, :], self.Phi_)
            self.W_[index_last_action, :] += self.learning_rate * delta * self.e

        self.t += 1
        self.last_action = action_
        self.last_observation = observation
        self.last_reward = reward

    def Phi_function(self, state):

        for i in range(self.state_bucket[0]):
            for j in range(self.state_bucket[1]):
                self.Phi_[i + self.state_bucket[0] * j] = (np.exp(-((state[0] - self.state_space[0][i]) / (self.state_bucket[0]))**2) * np.exp(-((state[1] - self.state_space[1][j]) / (self.state_bucket[1]))**2))

        return self.Phi_

    def Q_function(self, state, action):

        return np.dot(self.W_, self.Phi_function(state))




Agent = TDQAgent
