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

agent = ppad.Agent01()

observations_list, actions_list, rewards_list = ppad.smart_data(boards=16, permutations=3, trajectories=10, steps=-1)

agent.learn(observations=observations_list,
            actions=actions_list,
            rewards=rewards_list,
            iterations=10000,
            experience_replay=True,
            verbose=1)

