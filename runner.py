"""
This is the machinnery that runs your agent in an environment.

"""
import matplotlib.pyplot as plt
import numpy as np
import agent

class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

    def step(self):
        observation = self.environment.observe().clone()
        action = self.agent.act(observation).copy()
        (reward, done) = self.environment.act(action)
        self.agent.reward(observation, action, reward,done)
        return (observation, action, reward, done)

    def loop(self, games,nbr_epoch, max_iter):

        cumul_reward = 0.0
        list_cumul_reward=[]
        list_optimal_ratio = []
        list_aprox_ratio =[]

        for epoch_ in range(nbr_epoch):
            print(" -> epoch : "+str(epoch_))
            for g in range(1, games + 1):
                print(" -> games : "+str(g))
                for epoch in range(5):
                    self.environment.reset(g)
                    self.agent.reset(g)
                    cumul_reward = 0.0

                    for i in range(1, max_iter + 1):
                        # if self.verbose:
                        #   print("Simulation step {}:".format(i))
                        (obs, act, rew, done) = self.step()
                        cumul_reward += rew
                        if self.verbose:
                            # print(" ->       observation: {}".format(obs))
                            # print(" ->            action: {}".format(act))
                            # print(" ->            reward: {}".format(rew))
                            # print(" -> cumulative reward: {}".format(cumul_reward))
                            if done:
                                #solution from baseline algorithm
                                approx_sol =self.environment.get_approx()

                                #optimal solution
                                optimal_sol = self.environment.get_optimal_sol()

                                # print cumulative reward of one play, it is actually the solution found by the NN algorithm
                                print(" ->    Terminal event: cumulative rewards = {}".format(cumul_reward))

                                #print optimal solution
                                print(" ->    Optimal solution = {}".format(optimal_sol))

                                #we add in a list the solution found by the NN algorithm
                                list_cumul_reward.append(-cumul_reward)

                                #we add in a list the ratio between the NN solution and the optimal solution
                                list_optimal_ratio.append(cumul_reward/(optimal_sol))

                                #we add in a list the ratio between the NN solution and the baseline solution
                                list_aprox_ratio.append(cumul_reward/(approx_sol))

                        if done:
                            break
                np.savetxt('test_'+str(epoch_)+'.out', list_optimal_ratio, delimiter=',')
                np.savetxt('test_approx_' + str(epoch_) + '.out', list_aprox_ratio, delimiter=',')


            if self.verbose:
                print(" <=> Finished game number: {} <=>".format(g))
                print("")

        np.savetxt('test.out', list_cumul_reward, delimiter=',')
        np.savetxt('opt_set.out', list_optimal_ratio, delimiter=',')
        #plt.plot(list_cumul_reward)
        #plt.show()
        return cumul_reward

def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))

class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count)
        self.agents = iter_or_loopcall(agent_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [ False for _ in self.environments ]

    def game(self, max_iter):
        rewards = []
        for (agent, env) in zip(self.agents, self.environments):
            env.reset()
            agent.reset()
            game_reward = 0
            for i in range(1, max_iter+1):
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                game_reward += reward
                if stop :
                    break
            rewards.append(game_reward)
        return sum(rewards)/len(rewards)

    def loop(self, games,nb_epoch, max_iter):
        cum_avg_reward = 0.0
        for epoch in range(nb_epoch):
            for g in range(1, games+1):
                avg_reward = self.game(max_iter)
                cum_avg_reward += avg_reward
                if self.verbose:
                    print("Simulation game {}:".format(g))
                    print(" ->            average reward: {}".format(avg_reward))
                    print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward
