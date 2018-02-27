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

import numpy as np
import os

import ppad

# 1. Set up.

# Generate 1000 random episodes.
episodes = 1000
# On average it takes 80+ steps to solve the board. We set the number of steps
# per episode to 200 because we do random sampling. This number should probably
# even be higher than this.
steps = 10

env = ppad.pad()
agent = ppad.agent01()

observations_list = []
actions_list = []
discounted_rewards_list = []

# 2. Sampling.
for _ in range(episodes):
    env.reset()

    for _ in range(steps):
        action = env.action_space.sample()
        env.step(action)
    env.step('pass')
    discounted_rewards = ppad.discount(rewards=env.rewards, gamma=0.9, norm=True)

    # Keep the episode if there was any combo.
    if discounted_rewards[-1] > 0:
        observations_list.append(list(env.observations[:-2]))  # Don't need to save the end state.
        actions_list.append(list(env.actions[:-1]))            # Don't need to store 'pass'.
        discounted_rewards_list.append(list(discounted_rewards[:-1]))

# 3. Learning.
agent.learn(observations=observations_list,
            actions=actions_list,
            rewards=discounted_rewards_list,
            epochs=1)

# 4. Predict and visualization.
observation = env.reset(finger=np.array([2, 2]))
for _ in range(100):
    action = agent.action(observation)
    observation, _, _, _ = env.step(action=action)

print(env.actions)
env.episode2gif(path=os.environ['PYTHONPATH']+'/projects/agent_01.gif')