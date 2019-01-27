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
    # List of all player actions.
    player_action = ['up', 'down', 'left', 'right', 'pass']
    # Defining opposite actions that cancel each other's effect.
    opposite_actions = {'up': 'down',
                        'down': 'up',
                        'left': 'right',
                        'right': 'left',
                        'pass': 'pass'}

    def __init__(self, finger, dim_row, dim_col):
        self.previous_action = None
        self.finger = finger
        self.dim_row = dim_row
        self.dim_col = dim_col

    def sample(self, type='random', include_pass=False):
        num_action = len(self.player_action) - 1
        if include_pass:
            num_action += 1

        current_action = None
        valid_move = False

        if type == 'random':
            while not valid_move:
                # Attempt to take an action.
                current_action = self.player_action[random.randint(0, num_action-1)]
                valid_move = True

                # Do not take the exact reverse of the previous action.
                if self.opposite_actions[current_action] == self.previous_action:
                    valid_move = False
                elif self.finger[0] >= self.dim_row-1 and current_action == 'down':
                    valid_move = False
                elif self.finger[0] <= 0 and current_action == 'up':
                    valid_move = False
                elif self.finger[1] >= self.dim_col-1 and current_action == 'right':
                    valid_move = False
                elif self.finger[1] <= 0 and current_action == 'left':
                    valid_move = False
        else:
            raise Exception('ERROR: Unknown player action type!')

        return current_action
