import numpy as np
import torch


"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    def __init__(self, graph):
        self.graph_init=graph
        self.nodes=self.graph_init.nodes()
        self.observation=torch.zeros(1,self.nodes,1,dtype=torch.float)
        self.nbr_of_nodes=0

    def reset(self):
        self.observation = torch.zeros(1,self.nodes,1,dtype=torch.float)

    def observe(self):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        return self.observation

    def act(self,node):

        self.observation[:,node,:]=1
        reward=self.get_reward(self.observation)
        return reward

    def get_reward(self,observation):

        new_nbr_nodes=np.sum(observation[0].numpy())

        if new_nbr_nodes-self.nbr_of_nodes>0:
            reward=-1
        else:
            reward=0

        self.nbr_of_nodes=new_nbr_nodes

        #Minimum vertex set:


        for edge in self.graph_init.edges():
            if observation[:,edge[0],:]==0 and observation[:,edge[1],:]==0:
                done=False
                break
            else:
                done= True


        return (reward,done)

    def get_mvc_approx(self):
        cover_edge=[]
        edges= list(self.graph_init.edges())
        while len(edges)>0:
            edge = edges[np.random.choice(len(edges))]
            cover_edge.append(edge[0])
            cover_edge.append(edge[1])
            for edge_ in edges:
                if edge_[0]==edge[0] or edge_[0]==edge[1]:
                    edges.remove(edge_)
                else:
                    if edge_[1]==edge[1] or edge_[1]==edge[0]:
                        edges.remove(edge_)
        return len(cover_edge)


