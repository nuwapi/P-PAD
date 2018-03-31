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

    def __init__(self, finger, dim_h, dim_v):
        self.previous_action = None
        self.finger = finger
        self.dim_h = dim_h
        self.dim_v = dim_v

    def sample(self, type='random', include_pass=False):
        num_action = len(self.player_action) - 1
        if include_pass:
            num_action += 1

        current_action = None
        valid_move = False
        while not valid_move:
            current_action = self.player_action[random.randint(0, num_action-1)]
            current_set = {current_action, self.previous_action}

            valid_move = True
            # Do not take the exact reverse of the previous action.
            if current_set == {'left', 'right'} or current_set == {'up', 'down'}:
                valid_move = False
            elif self.finger[0] >= self.dim_h-1 and current_action is 'right':
                valid_move = False
            elif self.finger[0] <= 0 and current_action is 'left':
                valid_move = False
            elif self.finger[1] >= self.dim_v-1 and current_action is 'up':
                valid_move = False
            elif self.finger[1] <= 0 and current_action is 'down':
                valid_move = False

        return current_action
