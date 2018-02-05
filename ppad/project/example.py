"""
P-PAD
Copyright (C) 2018 NWP, CP

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

"""
# This example follows the same API of OpenAI Gym.

# Basic OpenAi Gym example.
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    # Take a random action.
    env.step(env.action_space.sample())

# OpenAi Gym used together with an agent.
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

# Inspect environment.
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
"""

import ppad.agent.someagent as someagent
import ppad.pad.game as game



# Initialization.
env = game.make()
# The agent should have an API that an take in a generic environment.
# Simulator should have all of the parameters needed for the game itself.
# Agent could have algorithm specific parameters.
# Agent should be able to take pre-trained policy.
agent = someagent(env=env, savf=None, )

#######################
# Learning.
n_episodes = 100
max_step = 10000
for i_episode in range(n_episodes):
    observation = env.reset()
    for i_step in range(max_step):
        env.render()
        print(observation)

        # 1. Take a random action.
        action = env.action_space.sample()
        # 2. Or take an action given by the current policy.
        action = agent.policy(state=env.state())

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(i_step + 1))
            break

#######################
# Analysis.
agent.stat()
agent.plot()

#######################
# Save results.
pad_policy = agent.policy()
pad_actionValues = agent.actionValues()
pad_stateValues = agent.stateValues()