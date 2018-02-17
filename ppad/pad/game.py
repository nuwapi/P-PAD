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
import random
from datetime import datetime

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
                 finger=None):

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
        random.seed(datetime.now())

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
        self.board = -1 * np.ones((self.dim_h, self.dim_v))
        # 0 means not locked, 1 means locked.
        self.locked = np.zeros((self.dim_h, self.dim_v))
        # 0 means not enhanced, 1 means enhanced.
        self.enhanced = np.zeros((self.dim_h, self.dim_v))
        # Information about player's team.
        self.team = team
        # Information about the enemy or enemies.
        self.enemy = enemy
        # Finger location of the board.
        if finger is None:
            self.finger = [0, 0]
            self.finger[0] = random.randint(0,dim_h-1)
            self.finger[1] = random.randint(0,dim_v-1)
        # The action space for this environment.
        self.action_space = player.PlayerAction()
        # The observation space for this environment.
        # TODO: Fill this in.
        self.observation_space = None
        # Coloring and letter scheme for rendering.
        self.render_dict = parm.default_render_dict

        self.fill_board()

    def step(self, action):
        reward = 0
        done = False
        if action == 'pass':
            done = True
            # A list of dictionaries representing combos.
            all_combos = []
            while True:
                combos = self.cancel()
                if len(combos) < 1:
                    break
                all_combos += combos
                self.drop()
                self.fill_board()
            reward = self.damage(all_combos)
        else:
            self.apply_action(action)

        # TODO: In the future, we would want to output locked and enhanced as observations as well.
        observation = self.board
        info = 'Currently, we only provide info.'

        return observation, reward, done, info

    def reset(self):
        random.seed(datetime.now())
        self.fill_board(reset=True)
        # Also reset finger location.
        return self.state()

    def render(self):
        print('+-----------+')
        for y in range(self.dim_v):
            reverse_y = self.dim_v - 1 - y
            if y > 0:
                print('|-----------|')
            print_str = '|'
            for x in range(self.dim_h):
                key = self.board[x, reverse_y]
                if x == self.finger[0] and reverse_y == self.finger[1]:
                    print_str += self.render_dict[key][0] + '|'
                else:
                    print_str += self.render_dict[key][1] + self.render_dict[key][0] + '\033[0m' + '|'
            print(print_str)
        print('+-----------+')

    def state(self):
        return self.board

    def fill_board(self, reset=False):
        """
        The state of the board is represented by integers.
        We do not handle locked and enhanced orbs at this stage.
        -1 - empty
        0  - red
        1  - blue
        2  - green
        3  - light
        4  - dark
        5  - heal
        6  - jammer
        7  - poison
        8  - mortal poison
        9  - bomb
        :param reset: If True, clear out the board first before filling.
        :return:
        """
        for orb in np.nditer(self.board, op_flags=['readwrite']):
            if orb == -1 or reset is True:
                orb[...] = self.random_orb(self.skyfall)

        # TODO: In the future, enhanced and locked arrays should also be updated here.

    def drop(self):
        for x in range(self.dim_h):
            col = []
            for y in range(self.dim_v):
                if self.board[x, y] != -1:
                    col.append(self.board[x, y])
                self.board[x, :] = -1
            for y in range(len(col)):
                self.board[x, y] = col[y]
        # TODO: Apply the same update to enhanced and locked orbs.

    def cancel(self):
        return []

    def apply_action(self, action):
        """
        :param action: left, right, up, down
        """
        if action == 'left':
            target_x = self.finger[0] - 1
            target_y = self.finger[1]
            if target_x >= 0:
                self.swap(self.finger[0], self.finger[1], target_x, target_y, True)
        elif action == 'right':
            target_x = self.finger[0] + 1
            target_y = self.finger[1]
            if target_x <= self.dim_h - 1:
                self.swap(self.finger[0], self.finger[1], target_x, target_y, True)
        elif action == 'up':
            target_x = self.finger[0]
            target_y = self.finger[1] + 1
            if target_y <= self.dim_v - 1:
                self.swap(self.finger[0], self.finger[1], target_x, target_y, True)
        elif action == 'down':
            target_x = self.finger[0]
            target_y = self.finger[1] - 1
            if target_y >= 0:
                self.swap(self.finger[0], self.finger[1], target_x, target_y, True)

    def swap(self, x1, y1, x2, y2, move_finger):
        """
        Swap the location of two orbs. x = 0 is bottom, y = 0 is left.
        Swap also update the finger location.
        :param x1: The horizontal coordinate of orb 1 (typically the one finger is one).
        :param y1: The vertical coordinate of orb 1.
        :param x2: The horizontal coordinate of orb 2.
        :param y2: The vertical coordinate of orb 2.
        :param move_finger: Whether to update finger location to the location of orb 2.
        """
        val1 = self.board[x1, y1]
        val2 = self.board[x2, y2]
        self.board[x1, y1] = val2
        self.board[x2, y2] = val1

        # val1 = self.enhanced[x1, y1]
        # val2 = self.enhanced[x2, y2]
        # self.enhanced[x1, y1] = val2
        # self.enhanced[x2, y2] = val1
        #
        # val1 = self.locked[x1, y1]
        # val2 = self.locked[x2, y2]
        # self.locked[x1, y1] = val2
        # self.locked[x2, y2] = val1

        if move_finger:
            self.finger[0] = x2
            self.finger[1] = y2

    def damage(self, combos):
        """
        Calculate team damage from a list of combos.
        """
        damage_tracker = {'blue':0,
                          'red': 0,
                          'green': 0,
                          'dark': 0,
                          'light': 0}
        # Calculate damage for each combo
        for combo in combos:
            color = combo['color']
            for unit in self.team: 
                # Find units with this color as main or sub element
                if unit['color_1']==unit['color_2']==color:
                    multiplier = 1.1
                elif  unit['color_1']==color:
                    multiplier = 1
                elif unit['color_2']==color:
                    multiplier = 0.3
                else:
                    continue
                # Multiplier from number of orbs
                multiplier *= 1+0.25*(combo['N']-3)
                # Multiplier from enhanced orbs
                # Multiplier from Two-Pronged Attack (TPA)
                # Multiplier from Void Damage Penetration (VDP)
                # Multiplier from L-shape
                # Multiplier from 7-combos
                # Multiplier from Heart Box
                # Final damage calculation for unit
                damage_tracker[color] += multiplier*unit['atk']
        
        # Modifiers at the color level
        for color in damage_tracker:
            # Multiplier from combos
            damage_tracker[color] *= 1+0.25*(len(combos)-1)
            # Multiplier from rows
            # Multiplier from enemy color
        # Final result        
        return sum(damage_tracker.values())

    @staticmethod
    def random_orb(prob):
        """
        :param prob: A list of non-negative numbers that sum up to 1.
        :return: A random integer in range [0, len(prob)-1] following the probability in prob.
        """
        random_number = random.random()
        prob_sum = 0
        for i in range(len(prob)):
            prob_sum += prob[i]
            if prob_sum >= random_number and prob[i] > 0:
                return i
        return len(prob)-1


def make():
    return Game()
