from unittest import result
import gym 
from collections import defaultdict
import numpy as np
import random
from bins import Discretizer
import pandas as pd
import matplotlib.pyplot as plt
import csv

BINS = 50
epsilon = 0.4
gamma = 0.95
alpha = 0.1
decay_rate = 0.99999


def select_action(obs):
    rand = random.uniform(0,1)
    if (rand<epsilon):
        return random.randint(0, 1)
    else:
        return state_action_values[(tuple(obs))].argmax()

def update_values(obs_t0, action, reward_t1, obs_t1):

    v_old = state_action_values[tuple(obs_t0)][action] 
    state_action_values[tuple(obs_t0)][action] = v_old + alpha*(reward_t1 + gamma*np.max(state_action_values[tuple(obs_t1)])-v_old)


if __name__=="__main__":
    env = gym.make('CartPole-v1')
    states = env.observation_space
    discretizer = Discretizer(states.low, states.high, BINS)
    state_action_values = defaultdict(lambda: np.zeros(env.action_space.n))

    result = {"i_episode": [],
                "Episodes": [], 
                "Reward": []}
        
    for i_episode in range(50000):
        observation = env.reset()
        accum_reward = 0
        state = discretizer.discretize(observation)

        for t in range(1000):
            env.render()
            action = select_action(state)
            observation, reward, done, info = env.step(action)
            accum_reward += reward
            next_state = discretizer.discretize(observation)
            epsilon = epsilon*decay_rate

            if done:
                result['i_episode'].append(i_episode)
                result['Episodes'].append(t+1)
                result['Reward'].append(accum_reward)
                state_action_values[tuple(state)][action] = -20


                if (i_episode%100==0):
                    print(f"{i_episode}: {np.mean(result['Episodes'][-100:])} , Epsilon: {epsilon}")
                break
            else:
                update_values(state, action, reward, next_state)


            state = next_state
    env.close()

    df = pd.DataFrame(result)
    df.plot.line("i_episode", "Episodes")
    plt.show()
    df.plot.line("i_episode", "Reward")
    plt.show()

    df.to_csv("results.csv", sep='\t', encoding='utf-8', columns=["i_episode", "Reward"])

    with open("state_action_values.csv", "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(state_action_values.keys())
        writer.writerows(zip(*state_action_values.values()))
