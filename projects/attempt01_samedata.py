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

import numpy as np

env = ppad.PAD()
agent = ppad.Agent01()

observations_list = []
actions_list = []
rewards_list = []

episodes = 2

init_board = np.copy(env.board)

action_sequence = ['left', 'right', 'up', 'down', 'left', 'right', 'up', 'down']
reward = np.array([10, 1, 1, 1, 1, 1, 1, 10])
finger_shift = [0, 0, 0, 0, 1, 1, 1, 1]

for i in range(len(action_sequence)):
    init_finger = np.array([2, 2])
    init_finger[0] += finger_shift[i]
    env.reset(board=init_board, finger=init_finger)
    env.step(action_sequence[i])
    env.rewards[0] = reward[i]
    observations_list.append(list(env.observations[:-1]))
    actions_list.append(list(env.actions))
    rewards_list.append(list(env.rewards))

for _ in range(10000):
    agent.learn(observations=observations_list,
                actions=actions_list,
                rewards=rewards_list,
                iterations=1,
                experience_replay=True,
                verbose=0)

    prediction = agent.model.predict(agent.convert_input(observations=observations_list)[0])
    diff = np.concatenate((prediction[0][0:4], prediction[4][0:4]), axis=0) - reward
    print(diff)
    print(np.sum(diff))
