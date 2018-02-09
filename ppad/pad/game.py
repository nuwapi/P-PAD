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
from ppad.pad.action.player import PlayerAction


class Game:
    # The sum of all 10 skyfall rates equals to 1.
    # In order, the 10 orb types are:
    #   [red,
    #    blue,
    #    green,
    #    light,
    #    dark,
    #    heal,
    #    jammer,
    #    poison,
    #    mortal poison,
    #    bomb]
    default_skyfall = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0, 0, 0, 0]

    def __init__(self,
                 dim_h=6,
                 dim_v=5,
                 skyfall=default_skyfall,
                 skyfall_modifier=None,
                 buff=None):
        """

        :param dim_h:
        :param dim_v:
        :param skyfall:
        :param skyfall_modifier:
        :param buff:
        """

        # Check the validity of the input parameters.
        if abs(sum(skyfall)-1) > 1e-4:
            raise Exception('ERROR: Skyfall rates should add up to 1!')
        for rate in skyfall:
            if rate < 0:
                raise Exception('ERROR: Skyfall rates should be a positive number!')
        if dim_h < 0 or dim_v < 0:
            raise Exception('ERROR: Board dimensions should be positive integers!')
        if dim_h*dim_v < 13:
            raise Exception('ERROR: The board should allow at least 13 orbs. Please increase board size!')

        # Initialize.
        self.dim_h = dim_h
        self.dim_v = dim_v
        self.skyfall = skyfall
        self.skyfall_modifier = skyfall_modifier
        self.buff = buff
        self.board = np.zeros((self.dim_h, self.dim_v))
        self.fill_board()
        self.action_space = PlayerAction()

    def step(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def state(self):
        pass

    def fill_board(self):
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

        :return:
        """
        pass


def make():
    return Game()
