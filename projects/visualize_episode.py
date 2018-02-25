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

env = ppad.make()
observation = env.reset()

starting_board = np.copy(observation[0])
starting_finger = np.copy(observation[1])
boards = []
actions = []
boards.append(starting_board)

for i in range(20):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    boards.append(np.copy(observation[0]))
    actions.append(action)

env.episode2gif(boards, actions, starting_finger, os.environ['PYTHONPATH']+'/assets/sample.gif')