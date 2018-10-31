"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""
import matplotlib.pyplot as plt
import numpy as np

class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

    def step(self):
        observation = self.environment.observe()
        action = self.agent.act(observation)
        (reward, done) = self.environment.act(action)
        self.agent.reward(observation, action, reward,done)
        return (observation, action, reward, done)

    def loop(self, games, max_iter):

        cumul_reward = 0.0
        list_cumul_reward_game=[]
        mean_reward = []

        for g in range(1, games+1):
            self.environment.reset()
            self.agent.reset()
            cumul_reward_game = 0.0

            for i in range(1, max_iter+1):
                #if self.verbose:
                    #print("Simulation step {}:".format(i))
                (obs, act, rew, done) = self.step()
                cumul_reward += rew
                cumul_reward_game+=rew
                if self.verbose:
                    #print(" ->       observation: {}".format(obs))
                    #print(" ->            action: {}".format(act))
                    #print(" ->            reward: {}".format(rew))
                    #print(" -> cumulative reward: {}".format(cumul_reward))
                    if done:
                        print(" ->    Terminal event: cumulative rewards = {}".format(cumul_reward_game))
                        print(" ->    MVC_approx = {}".format(self.environment.get_mvc_approx()))

                        list_cumul_reward_game.append(-cumul_reward_game)
                        if g>100:
                            mean_reward.append(np.mean(list_cumul_reward_game[-100:]))
                if done:
                    break


            if self.verbose:
                print(" <=> Finished game number: {} <=>".format(g))
                print("")
        plt.plot(mean_reward)
        plt.show()
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
                if stop =="stop":
                    break
            rewards.append(game_reward)
        return sum(rewards)/len(rewards)

    def loop(self, games, max_iter):
        cum_avg_reward = 0.0
        for g in range(1, games+1):
            avg_reward = self.game(max_iter)
            cum_avg_reward += avg_reward
            if self.verbose:
                print("Simulation game {}:".format(g))
                print(" ->            average reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward
