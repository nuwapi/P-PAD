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

import ppad

# 1. Set up.

# Generate 100 random episodes.
episodes = 10
# On average it takes 80+ steps to solve the board. We set the number of steps
# per episode to 200 because we do random sampling. This number should probably
# even be higher than this.
steps = 10

env = ppad.PAD()
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

    # Keep the episode if there was any combo.
    if env.rewards[-1] > 0:
        discounted_rewards = ppad.discount(rewards=env.rewards, gamma=0.9)
        observations_list.append(list(env.observations))  # Don't need to save the end state.
        actions_list.append(list(env.actions))
        discounted_rewards_list.append(list(discounted_rewards))

# 3. Learning.
agent.learn(observations=observations_list,
            actions=actions_list,
            rewards=discounted_rewards_list,
            iterations=1,
            experience_replay=False)

# 4. Predict and visualization.
for _ in range(100):
    observation = env.reset()
    for _ in range(2):
        action = agent.action(observation)
        observation, _, _, _ = env.step(action=action)
    print(env.actions)
