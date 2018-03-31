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
from ppad.pad.utils import episode2gif

# Example 1: Visualize directly from the environment itself.
env = ppad.PAD()

for _ in range(100):
    env.step(action=env.action_space.sample())

env.visualize(path='YOURPATH/random_sampling.gif')

# Example 2: Visualize using the episode information.
observations, actions, rewards = ppad.smart_data(boards=1, permutations=1, trajectories=1, steps=100)
episode2gif(observations, actions, path='YOURPATH/smart_data.gif')
