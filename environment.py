import numpy as np
import graph


"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    def __init__(self, graph_type: str):
        self.graph_init=graph.Graph(graph_type,n,p,m)
        self.nodes=self.graph_init.nodes()
        self.observation=np.zeros(self.nodes)
        self.nbr_of_nodes=0

    def reset(self):
        self.observation = np.zeros(self.nodes)

    def observe(self):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        return self.observation

    def act(self,node):
        self.observation[node]=1
        reward=self.get_reward(self.observation)
        return reward

    def get_reward(self,observation):

        new_nbr_nodes=np.sum(observation)

        if new_nbr_nodes-self.nbr_of_nodes>0:
            reward=-1
        else:
            reward=0

        self.nbr_of_nodes=new_nbr_nodes

        #Minimum vertex set:


        for edge in self.graph_init.edges:
            if observation[edge[0]]==0 and observation[edge[1]]==0:
                stop= "no_stop"
                break
            else:
                stop= "stop"


        return (reward,stop)



# class Environment:
#     def __init__(self, g=10.0, d=100.0, H=10., m=10.0, F=3.0):
#         """Instanciate a new environement in its initial state.
#         """
#         self.mc = MountainCar(g=g, d=d, H=H, m=m, F=F, R=1.0, T=0.0)
#
#     def reset(self):
#         self.mc.reset()
#         # place the car at a random place near the bottom
#         self.mc.x = np.random.uniform(-self.mc.d*1.3, -self.mc.d*0.7)
#
#     def get_range(self):
#         return [-1.5*self.mc.d, 0.0]
#
#     def observe(self):
#         """Returns the current observation that the agent can make
#         of the environment, if applicable.
#         """
#         return (self.mc.x, self.mc.vx)
#
#     def act(self, action):
#         """Perform given action by the agent on the environment,
#         and returns a reward.
#         """
#         reward = self.mc.act(action)
#         return (reward, "victory" if reward > 0.0 else None)
