import argparse
import agent
import environment
import runner
import graph
import logging
import networkx as nx
import sys

# # 2to3 compatibility
# try:
#     input = raw_input
# except NameError:
#     pass

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment', metavar='ENV_CLASS', type=str, default='Environment', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--graph_type',metavar='GRAPH', default='erdos_renyi',help ='Type of graph to optimize')
parser.add_argument('--graph_nbr', type=int, default='1000', help='number of graph to generate')
parser.add_argument('--model', type=str, default='S2V_QN', help='model name')
parser.add_argument('--ngames', type=int, metavar='n', default='999', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per game')
parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')
parser.add_argument('--interactive', action='store_true', help='After training, play once in interactive mode. Ignored in batch mode.')

def main():
    args = parser.parse_args()
    # agent_class = eval('agent.{}'.format(args.agent))
    # env_class = eval('environment.{}({})'.format(args.environment,args.graph))
    logging.info('Loading graph...')
    graph_dic = {}
    for graph_ in range(args.graph_nbr):
        graph_dic[graph_] = graph.Graph(graph_type=args.graph_type, cur_n=50, p=.09)

    logging.info('Loading agent...')
    agent_class = agent.Agent(graph_dic, args.model)

    logging.info('Loading environment...')
    env_class = environment.Environment(graph_dic)

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        my_runner = runner.BatchRunner(env_class, agent_class, args.batch, args.verbose)
        final_reward = my_runner.loop(args.ngames, args.niter)
        print("Obtained a final average reward of {}".format(final_reward))
    else:
        print("Running a single instance simulation...")
        my_runner = runner.Runner(env_class, agent_class, args.verbose)
        final_reward = my_runner.loop(args.ngames, args.niter)
        print("Obtained a final reward of {}".format(final_reward))





        # if args.interactive:
        #     import pylab as plb
        #     from mountaincarviewer import MountainCarViewer
        #
        #     # prepare for the visualization
        #     plb.ion()
        #     env = my_runner.environment
        #     a = my_runner.agent
        #     env.reset()
        #     a.reset(env.get_range())
        #     mv = MountainCarViewer(env.mc)
        #     mv.create_figure(args.niter, args.niter)
        #     plb.draw()
        #
        #
        #     for _ in range(args.niter):
        #         print('\rt = {}'.format(env.mc.t))
        #         print("Enter to continue...")
        #         input()
        #
        #         sys.stdout.flush()
        #
        #         obs = env.observe()
        #         action = a.act(obs)
        #         (reward, stop) = env.act(action)
        #         a.reward(obs, action, reward)
        #
        #         # update the visualization
        #         mv.update_figure()
        #         plb.draw()
        #
        #         # check for rewards
        #         if stop is not None:
        #             print("\rTop reached at t = {}".format(env.mc.t))
        #             break

if __name__ == "__main__":
    main()
