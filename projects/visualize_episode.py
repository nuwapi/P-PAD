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

from importlib import reload
import ppad
ppad = reload(ppad)
from ppad.pad.utils import episode2gif


SOMEPATH = 'yourpath'

# Example 1: Visualize directly from the environment itself.
env = ppad.PAD()
for _ in range(100):
    env.step(action=env.action_space.sample(), verbose=True)
env.visualize(filename=SOMEPATH + '/random_sampling.gif')

env.step(action='pass', verbose=True)

# Example 2: Visualize using the episode information.
# Generate observations and actions using any method in the specified format.
# Here we are generating them from "smart data" and step = -1 means terminate on zero combo.
observations, actions, rewards = ppad.smart_data(boards=1, permutations=1, trajectories=1, steps=-1)
episode2gif(observations, actions, filename=SOMEPATH + '/smart_data.gif')
