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

import os

import ppad

# 1. Set up.

# Generate 1000 random episodes.
episodes = 1000
# On average it takes 80+ steps to solve the board. We set the number of steps
# per episode to 200 because we do random sampling. This number should probably
# even be higher than this.
steps = 200

env = ppad.make()
agent = ppad.agent01()

# 2. Sampling.
for _ in range(episodes):
    env.reset()
    for j in range(steps):
        action = env.action_space.sample()
        if j >= steps-1:
            action = 'pass'
        env.step(action)

    ppad.discount(env.rewards)

    # 3. Learning.
    agent.learn(observations=env.observations, rewards=env.rewards)

# 4. Predict and visualization.
observation = env.reset()
for _ in range(100):
    action = agent.predict(observation)
    observation, _, _, _ = env.step(action=action)
env.episode2gif(path=os.environ['PYTHONPATH']+'/projects/agent_01.gif')