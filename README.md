# Implementation of Dai and al. paper (Learning combinatorial optimization algorithms over graph) in Python and torch
Source : https://arxiv.org/abs/1704.01665


## Introduction

The goal of this study is to implement the work done by Dai and al. who developed a uniﬁed framework to solve combinatorial optimization problem over graph through learning. The model is trained on batch of graphs from a `Erdös-Rényi` and a `Barabási–Albert` degree distribution, with a ﬁx number of node. Firstly, the algorithm learns the structure of the graph through a graph embedding algorithm. A helping function provides as an input the current state of progress of the optimisation. Then it chooses the best node to add to the optimal set through a reinforcement learning algorithm. I have worked on two combinatorial optimization problems, `Minimum Vertex Cover (MVC) ` and `Maximum Cut Set (MAXCUT)`.

## Structure of the code


![alt text](https://raw.githubusercontent.com/louisv123/COLGE/master/utils/structure.png)

### main.py

`main.py` allows to define arguments and launch the code.

Arguments : 

- `--environment_name`, type=str, default='MVC' : environment_name is the kind of optimization problem. It must be `MVC` for Minimum Vertex Cover problem or `MAXCUT` for Maximum Cut Set problem.
    
- `--agent`, type=str,  default='Agent' : Define the name of the agent.

- `--graph_type`, type=str, default='erdos_renyi' : define the kind of degree distribution of graphs. It must be among `erdos_renyi`, `powerlaw`, `barabasi_albert` or `gnp_random_graph`.

- `--graph_nbr`, type=int, default='1000', : number of graph to generate.

- `--model`, type=str, default='GCN_QN_1', model name for Q-function. It must be either `S2V_QN_1` for structure2vec algorithm or `GCN_QN_1` for graph_convolunional network algortihm.

- `--ngames`, type=int, default='500': number of games to simulate per epochs.

- `--niter`, type=int, default='1000', max number of iterations per game if the algorithm doesn't reach the terminal step.

- `--batch`, type=int, default=None, batch run several agent at the same time.

### graph.py

Define the graph object, espacially the kind of degree distribution and the methods.

### runner.py 

This script calls each step of reinforcement learning part in a loop (epochs + games):
  - `observe`: get states features of the problem (`environment class`)
  - `act`: take an action from the last observation (`agent class`)
  - get `reward` and `done` information from the last action (`environment class`)
  - perform a learning step with last observation, last action, observation and reward  (`agent class`)
  
### agent.py

Define the agent object and methods needed in deep Q-learning algorithm 

### model.py

Define the Q-function and the embedding algorithm. `S2V_QN_1` and `GCN_QN_1` gives good results.

### environment.py

Define the environment object which is either MVC (Minimum vertex cover) or MAXCUT (Maximum cut set).
It contains as well the method to get the approximation solution and the optimal solution (with pulp)

  


