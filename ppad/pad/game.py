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

from datetime import datetime
import numpy as np
import random

import ppad.pad.parameters as parameters
import ppad.pad.player as player
import ppad.pad.utils as pad_utils


class PAD:
    def __init__(self,
                 dim_h=6,
                 dim_v=5,
                 skyfall=parameters.default_skyfall,
                 skyfall_locked=parameters.default_skyfall_locked,
                 skyfall_enhanced=parameters.default_skyfall_enhanced,
                 skyfall_damage=True,
                 buff=None,
                 team=parameters.default_team,
                 enemy=parameters.default_enemy,
                 board=None,
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
        # Seed random number generator.
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
        # Whether to skyfall when calculating damange.
        self.skyfall_damange = skyfall_damage
        # All player and enemy buffs and debuffs. This should be a list of some sort.
        self.buff = buff
        # The orb identities of the board. See fill_board() for details.
        self.board = -1 * np.ones((self.dim_h, self.dim_v), dtype=int)
        # 0 means not locked, 1 means locked.
        self.locked = np.zeros((self.dim_h, self.dim_v), dtype=int)
        # 0 means not enhanced, 1 means enhanced.
        self.enhanced = np.zeros((self.dim_h, self.dim_v), dtype=int)
        # Finger numpy array.
        self.finger = np.zeros(2, dtype=int)
        # Information about player's team.
        self.team = team
        # Information about the enemy or enemies.
        self.enemy = enemy
        # The observation space for this environment.
        # TODO: Fill this in.
        self.observation_space = None
        # Coloring scheme and letter scheme for simple "rendering" in the terminal.
        self.render_dict = parameters.default_render_dict

        # The following five variables are used to store the game state sequence for the current episode.
        # List of dim_h by dim_v numpy arrays.
        self.boards = []
        # List of 2 by 1 numpy arrays.
        self.fingers = []
        # List of (board, finger) tuples. See above for board and finger definitions.
        self.observations = []
        # List of integers.
        self.actions = []
        # List of floats.
        self.rewards = []
        # The action space for this environment.
        self.action_space = player.PlayerAction(finger=self.finger, dim_h=self.dim_h, dim_v=self.dim_v)

        # Initialize game state.
        self.reset(board=board, finger=finger)

    def apply_action(self, action):
        """
        Swap two orbs on all boards (specified in self.swap) given the action.
        :param action: left, right, up, down
        """
        if action == 'left':
            target_x = self.finger[0] - 1
            target_y = self.finger[1]
            if target_x >= 0:
                self.swap(self.finger[0], self.finger[1], target_x, target_y, True)
            else:
                return 'rejected'
        elif action == 'right':
            target_x = self.finger[0] + 1
            target_y = self.finger[1]
            if target_x <= self.dim_h - 1:
                self.swap(self.finger[0], self.finger[1], target_x, target_y, True)
            else:
                return 'rejected'
        elif action == 'up':
            target_x = self.finger[0]
            target_y = self.finger[1] + 1
            if target_y <= self.dim_v - 1:
                self.swap(self.finger[0], self.finger[1], target_x, target_y, True)
            else:
                return 'rejected'
        elif action == 'down':
            target_x = self.finger[0]
            target_y = self.finger[1] - 1
            if target_y >= 0:
                self.swap(self.finger[0], self.finger[1], target_x, target_y, True)
            else:
                return 'rejected'
        return 'accepted'

    def damage(self, combos):
        """
        Calculate team damage from a list of combos.
        :param: combos: The data structure of this list can be found in cancel().
        :return: Total damage.
        """
        damage_tracker = {'blue': 0,
                          'red': 0,
                          'green': 0,
                          'dark': 0,
                          'light': 0}
        # Calculate damage for each combo
        for combo in combos:
            color = combo['color']
            for unit in self.team:
                # Find units with this color as main or sub element
                if unit['color_1'] == unit['color_2'] == color:
                    multiplier = 1.1
                elif unit['color_1'] == color:
                    multiplier = 1
                elif unit['color_2'] == color:
                    multiplier = 0.3
                else:
                    continue
                # Multiplier from number of orbs
                multiplier *= 1 + 0.25 * (combo['N'] - 3)
                # Multiplier from enhanced orbs
                # Multiplier from Two-Pronged Attack (TPA)
                # Multiplier from Void Damage Penetration (VDP)
                # Multiplier from L-shape
                # Multiplier from 7-combos
                # Multiplier from Heart Box
                # Final damage calculation for unit
                damage_tracker[color] += multiplier * unit['atk']

        # Modifiers at the color level
        for color in damage_tracker:
            # Multiplier from combos
            damage_tracker[color] *= 1 + 0.25 * (len(combos) - 1)
            # Multiplier from rows
            # Multiplier from enemy color
        # Final result
        return sum(damage_tracker.values())

    def drop(self):
        """
        Move all orbs vertically downwards to fill in empty spaces.
        """
        for x in range(self.dim_h):
            col = []
            for y in range(self.dim_v):
                if self.board[x, y] != -1:
                    col.append(self.board[x, y])
            self.board[x, :] = -1
            for y in range(len(col)):
                self.board[x, y] = col[y]
        # TODO: Apply the same update to enhanced and locked boards.

    def visualize(self, filename, shrink=3, animate=True):
        """
        Note: One needs to have PPADPATH defined to be the root directory (P-PAD) of this repo.
        :param filename: The location where intermediate pngs and the final gif are stored.
        :param shrink: Shrink the output image by this many times along each dimension.
        :param animate: If true, output an animated GIF, it false, output the PNG of the first frame.
        """
        pad_utils.episode2gif(observations=self.observations, actions=self.actions, filename=filename, shrink=shrink, animate=animate)

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
        """
        for orb in np.nditer(self.board, op_flags=['readwrite']):
            if orb == -1 or reset is True:
                orb[...] = self.random_orb(self.skyfall)

        # TODO: In the future, enhanced and locked arrays should also be updated here.

    def render(self, board=None):
        """
        Print a simple colored board to the terminal.
        :param board: One can input any Numpy array for rendering. It uses self.board by default.
        """
        if board is None:
            board = self.board

        print('+-----------+')
        for y in range(self.dim_v):
            reverse_y = self.dim_v - 1 - y
            if y > 0:
                print('|-----------|')
            print_str = '|'
            for x in range(self.dim_h):
                key = board[x, reverse_y]
                if x == self.finger[0] and reverse_y == self.finger[1]:
                    print_str += self.render_dict[key][0] + '|'
                else:
                    print_str += self.render_dict[key][1] + self.render_dict[key][0] + '\033[0m' + '|'
            print(print_str)
        print('+-----------+')

    def reset(self, board=None, finger=None):
        """
        Reset random number generator and game state.
        :return: The state of the system.
        """
        # Delete the complete record of the episode.
        del self.observations
        del self.boards
        del self.fingers
        del self.actions
        del self.rewards

        self.observations = []
        self.boards = []
        self.fingers = []
        self.actions = []
        self.rewards = []
        self.action_space.previous_action = None

        # Reset game state.
        random.seed(datetime.now())
        if board is None:
            while True:
                self.fill_board(reset=True)
                combos = pad_utils.cancel(self.board)
                if len(combos) == 0:
                    break
        else:
            self.board = np.copy(board)
        if finger is None:
            self.finger[0] = random.randint(0, self.dim_h - 1)
            self.finger[1] = random.randint(0, self.dim_v - 1)
        else:
            self.finger = np.copy(finger)

        # Append starting state to the game record.
        self.observations.append((np.copy(self.board), np.copy(self.finger)))

        self.boards.append(self.observations[-1][0])
        self.fingers.append(self.observations[-1][1])

        return self.state()

    def state(self):
        # TODO: Need to fill in this function properly.
        return self.boards[-1], self.fingers[-1]

    def step(self, action, verbose=False):
        """
        Step function comply to OpenAI Gym standard.
        :param action: The action to take in this step.
        :param verbose: Print intermediate board state and combos list.
        :return: See Gym documentation for details.
        """
        # We don't give intermediate rewords.
        reward = 0
        # The finger moving episode is not over by default.
        done = False
        # Make sure the PlayerAction finger position is up to date.
        self.action_space.finger = self.finger
        # If the agent decides to stop moving the finger.
        if action is 'pass':
            done = True
            # A list of dictionaries to store combos.
            all_combos = []
            # Repeat the combo detection until nothing more can be canceled.
            while True:
                combos = pad_utils.cancel(self.board)
                if verbose is True:
                    print('Board after combo canceling:')
                    self.render()

                # Break out of the loop if nothing can be canceled.
                if len(combos) < 1:
                    break

                # Add combo to combo list and skyfall.
                all_combos += combos
                if self.skyfall_damange is False:
                    break

                self.drop()
                if verbose is True:
                    print('Board after orb drop:')
                    self.render()
                self.fill_board()
                if verbose is True:
                    print('Board after fill board:')
                    self.render()

            # Reward is the total damage calculated based on combos.
            reward = self.damage(all_combos)
            if verbose is True:
                print(all_combos)
                print('The total damange is:', reward)
        else:
            comment = self.apply_action(action)
            if comment == 'accepted':
                self.action_space.previous_action = action
            else:
                print('Invalid move, you cannot move off the board!')

        # TODO: In the future, we would want to output locked and enhanced as observations as well.
        info = 'Currently, we do not provide info.'

        # Save current state to the record.
        if action is not 'pass':
            self.observations.append((np.copy(self.board), np.copy(self.finger)))
            self.boards.append(self.observations[-1][0])
            self.fingers.append(self.observations[-1][1])
        self.actions.append(action)
        self.rewards.append(reward)

        return (self.boards[-1], self.fingers[-1]), reward, done, info

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

        # val1 = self.locked[x1, y1]
        # val2 = self.locked[x2, y2]
        # self.locked[x1, y1] = val2
        # self.locked[x2, y2] = val1

        if move_finger:
            self.finger[0] = x2
            self.finger[1] = y2

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
