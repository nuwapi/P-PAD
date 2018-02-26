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

import random


class PlayerAction:
    # List of player actions.
    player_action = ['up', 'down', 'left', 'right', 'pass']

    def __init__(self):
        self.previous_action = None

    def sample(self, type='random', include_pass=False):
        num_action = len(self.player_action) - 1
        if include_pass:
            num_action += 1

        while True:
            current_action = self.player_action[random.randint(0, num_action-1)]
            current_set = {current_action, self.previous_action}
            # Do not take the exact reverse of the previous action.
            if current_set != {'left', 'right'} and current_set != {'up', 'down'}:
                return current_action
