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

import ppad.pad.parm as parm
import ppad.pad.action.player as player

class Game:
    def __init__(self,
                 dim_h=6,
                 dim_v=5,
                 skyfall=parm.default_skyfall,
                 skyfall_locked=parm.default_skyfall_locked,
                 skyfall_enhanced=parm.default_skyfall_enhanced,
                 buff=None,
                 team=parm.default_team,
                 enemy=parm.default_enemy,
                 finger=parm.default_finger):

        # Check the validity of the input parameters.
        # TODO: Add more parameter checks.
        if abs(sum(skyfall)-1) > 1e-4:
            raise Exception('ERROR: Skyfall rates should add up to 1!')
        for prob in skyfall:
            if prob < 0:
                raise Exception('ERROR: Skyfall rates should be a positive number!')
        for prob in skyfall_locked:
            if prob < 0 or prob > 1:
                raise Exception('ERROR: Locked orb probability is between 0 and 1!')
        for prob in skyfall_enhanced:
            if prob < 0 or prob > 1:
                raise Exception('ERROR: Enhanced orb probability is between 0 and 1!')
        if dim_h < 0 or dim_v < 0:
            raise Exception('ERROR: Board dimensions should be positive integers!')
        if dim_h*dim_v < 13:
            raise Exception('ERROR: The board should allow at least 13 orbs. Please increase board size!')

        # Initialize.
        # The horizontal length of the board.
        self.dim_h = dim_h
        # The vertical length of the board.
        self.dim_v = dim_v
        # The skyfall probabilities of different orb types.
        self.skyfall = skyfall
        # The locked orb probability for each orb type.
        self.skyfall_locked = skyfall_locked
        # The enhanced orb probability for each orb type.
        self.skyfall_enhanced = skyfall_enhanced
        # All player and enemy buffs and debuffs. This should be a list of some sort.
        self.buff = buff
        # The orb identities of the board. See fill_board() for details.
        self.board = np.zeros((self.dim_h, self.dim_v))
        # 0 means not locked, 1 means locked.
        self.locked = np.zeros((self.dim_h, self.dim_v))
        # 0 means not enhanced, 1 means enhanced.
        self.enhanced = np.zeros((self.dim_h, self.dim_v))
        # Information about player's team.
        self.team = team
        # Information about the enemy or enemies.
        self.enemy = enemy
        # Finger location of the board.
        self.finger = finger
        # The action space for this environment.
        self.action_space = player.PlayerAction()
        # The observation space for this environment.
        # TODO: Fill this in.
        self.observation_space = None

        self.fill_board()

    def step(self, action):
        reward = 0
        done = False
        if action == 'pass':
            done = True
            # A list that stores orb type and number of orbs in 2-tuple.
            # TODO: Expand to a 4-tuple.
            all_canceled = []
            while True:
                canceled = self.cancel()
                if len(canceled) < 1:
                    break
                all_canceled += canceled
                self.drop()
                self.fill_board()
            reward = self.damage(all_canceled)
        else:
            self.apply_action(action)

        # TODO: In the future, we would want to output locked and enhanced as observations as well.
        observation = self.board
        info = 'Currently, we only provide info.'

        return observation, reward, done, info

    def reset(self):
        self.fill_board(reset=True)
        # Also reset finger location.
        return self.state()

    def render(self):
        print('The render function has not been implemented.')

    def state(self):
        return self.board

    def fill_board(self, reset=False):
        """
        The state of the board is represented by integers.
        0  - empty
        1  - red
        2  - blue
        3  - green
        4  - light
        5  - dark
        6  - heal
        7  - jammer
        8  - poison
        9  - mortal poison
        10 - bomb

        :param reset: If True, clear out the board first before filling.
        :return:
        """
        pass

    def drop(self):
        pass

    def cancel(self):
        return []

    def apply_action(self, action):
        # Move orb and update finger location.
        pass

    def damage(self, cancels):
        return 0


def make():
    return Game()
