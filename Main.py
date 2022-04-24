import time
from ActorCritic import *
from QLearning import *
from ExpectedSARSA import *
from EmpireTakeoverEnv import *
from EmpireTakeoverEnv_v2 import *

import pygame
import matplotlib.pyplot as plt


#### Load the Map1 environment ####
env = ETEnv()
#### Load the Map2 environment ####
# env = ETEnvV2()

#### Expected-SARSA method ####
agent = EpsilonGreedyTabularExpectedSARSA(env=env, gamma=0.9, decay=.999,
          decay_type="exponential", min_epsilon=.001, epsilon=0.15,
          min_alpha=0.01, alpha=0.3, return_vals="progress")

returns, avg = agent.learn_task(
    num_episodes=500, decay_alpha=False, decay_epsilon=False)

plt.plot(avg)
plt.show()
print(avg)

env.init_render()
agent.play(epsilon=0.15, alpha=0.3, decay_alpha=False, decay_epsilon=True)

#### Q-Learning method ####
# agent = EpsilonGreedyTabularQLearning(env = env, gamma = 0.9, decay = .999,
#           decay_type = "exponential", min_epsilon = .05, epsilon = 0.15,
#           min_alpha = 0.01, alpha = 0.3, return_vals = "progress")
#
# returns, avg = agent.learn_task(
#     num_episodes = 500, decay_alpha = False, decay_epsilon = False)
#
# plt.plot(avg)
# plt.show()
# print(avg)
#
# env.init_render()
# agent.play(epsilon=0.15, alpha=0.3, decay_alpha=False, decay_epsilon=True)


#### The Actor-Critic method ####
# env.init_render()
# agent = ActorCritic_Batch(env, .0001, .0001, .9)
# rew, avg = agent.learn_task(n_episodes=1000)
#
# pygame.quit()
# print(rew)
# print(avg)
#
# env.init_render()
# agent.play()

time.sleep(10)
