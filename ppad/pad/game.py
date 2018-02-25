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
import os
from PIL import Image
import random

import ppad.pad.parm as parm
import ppad.pad.player as player


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
        # All player and enemy buffs and debuffs. This should be a list of some sort.
        self.buff = buff
        # The orb identities of the board. See fill_board() for details.
        self.board = -1 * np.ones((self.dim_h, self.dim_v), dtype=int)
        # 0 means not locked, 1 means locked.
        self.locked = np.zeros((self.dim_h, self.dim_v), dtype=int)
        # 0 means not enhanced, 1 means enhanced.
        self.enhanced = np.zeros((self.dim_h, self.dim_v), dtype=int)
        # Information about player's team.
        self.team = team
        # Information about the enemy or enemies.
        self.enemy = enemy
        # Finger location of the board. If no location is given, choose a random location.
        if finger is None:
            self.finger = np.zeros(2, dtype=int)
            self.finger[0] = random.randint(0, dim_h-1)
            self.finger[1] = random.randint(0, dim_v-1)
        # The action space for this environment.
        self.action_space = player.PlayerAction()
        # The observation space for this environment.
        # TODO: Fill this in.
        self.observation_space = None
        # Coloring scheme and letter scheme for simple "rendering" in the terminal.
        self.render_dict = parm.default_render_dict

        self.fill_board()

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

    def cancel(self):
        """
        Cancel all 3+ connected orbs and generate combos.
        """
        combos = []
        for i in range(self.dim_h):
            for j in range(self.dim_v):
                # If this orb has not been checked.
                orb_type = self.board[i, j]
                if orb_type != -1:
                    # 1. Detect island.
                    # Initially, the island only contains the starting orb.
                    island = np.zeros((self.dim_h, self.dim_v))
                    # Detect island starting from position i, j and update island array.
                    self.detect_island(self.board, island, i, j, orb_type)
                    # 2. Prune detected island.
                    pruned_island = np.zeros((self.dim_h, self.dim_v))
                    for k in range(self.dim_h - parm.c + 1):
                        for l in range(self.dim_v):
                            if np.sum(island[k:k+3, l]) == 3:
                                pruned_island[k:k+3, l] = 1
                    for k in range(self.dim_h):
                        for l in range(self.dim_v - parm.c + 1):
                            if np.sum(island[k, l:l+3]) == 3:
                                pruned_island[k, l:l+3] = 1
                    # 3. Save the combo and update board.
                    count = np.sum(pruned_island)
                    # If there is a combo.
                    if count >= parm.c:
                        # Update board.
                        for index, element in np.ndenumerate(pruned_island):
                            if element == 1:
                                self.board[index] = -1
                        # Generate combo.
                        # TODO: Add shape detection and enhanced orb count in the future.
                        combo = {'color': parm.int2english[orb_type],
                                 'N': count,
                                 'enhanced': None,
                                 'shape': None}
                        combos.append(combo)
        return combos

    def damage(self, combos):
        """
        Calculate team damage from a list of combos.
        :param: combos: The data structure of this list can be found in self.cancel().
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

    def detect_island(self, board, island, x, y, orb_type):
        """
        Recursively detect islands of conneted orbs.
        :param board: A Numpy array with self.dim_h and self.dim_v in size.
        :param island: A Numpy array of the same size as board, initially set to all zeroes.
        :param x: The starting search location.
        :param y: The starting search location.
        :param orb_type: Search for an island of this orb type.
        """
        if board[x, y] == orb_type:
            island[x, y] = 1
            # Go up.
            if y+1 < self.dim_v and island[x, y+1] == 0:
                self.detect_island(board, island, x, y+1, orb_type)
            # Go down.
            if y-1 >= 0 and island[x, y-1] == 0:
                self.detect_island(board, island, x, y-1, orb_type)
            # Go left.
            if x+1 < self.dim_h and island[x+1, y] == 0:
                self.detect_island(board, island, x+1, y, orb_type)
            # Go right.
            if x-1 >= 0 and island[x-1, y] == 0:
                self.detect_island(board, island, x-1, y, orb_type)

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

    def reset(self):
        """
        Reset random number generator and the whole board.
        :return: The state of the system.
        """
        random.seed(datetime.now())
        self.fill_board(reset=True)
        # Also reset finger location.
        return self.state()

    def state(self):
        # TODO: Need to fill in this function properly.
        return np.copy(self.board), np.copy(self.finger)

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
        # If the agent decides to stop moving the finger.
        if action == 'pass':
            done = True
            # A list of dictionaries to store combos.
            all_combos = []
            # Repeat the combo detection until nothing more can be canceled.
            while True:
                combos = self.cancel()
                if verbose is True:
                    print('Board after combo canceling:')
                    self.render()

                # Break out of the loop if nothing can be canceled.
                if len(combos) < 1:
                    break

                # Add combo to combo list and skyfall.
                all_combos += combos
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
            self.apply_action(action)

        # TODO: In the future, we would want to output locked and enhanced as observations as well.
        info = 'Currently, we do not provide info.'

        return (np.copy(self.board), np.copy(self.finger)), reward, done, info

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
    def episode2gif(boards, actions, starting_position, path):
        """
        Note: One needs to have PYTHONPATH defined to be the root directory of this repo.
        :param boards: A list of Numpy arrays representing the board at each step of an episode.
        :param actions: A list of actions (in English). actions[k] changes episode[k] to episode[k+1].
        :param starting_position: The finger [x, y] position at the start, i.e. where the action starts.
        :param path: The location where intermediate pngs and the final gif are stored.
        """
        asset_path = os.environ['PYTHONPATH'] + '/assets/'
        orbs = [asset_path + 'red.png',
                asset_path + 'blue.png',
                asset_path + 'green.png',
                asset_path + 'light.png',
                asset_path + 'dark.png',
                asset_path + 'heal.png',
                asset_path + 'jammer.png',
                asset_path + 'poison.png',
                asset_path + 'mortal_poison.png',
                asset_path + 'bomb.png']
        bg = [asset_path + 'bg1.png', asset_path + 'bg2.png']
        finger = starting_position
        dim_h = boards[0].shape[0]
        dim_v = boards[0].shape[1]
        # The length of the square image of the orbs is 100 px.
        edge = 100
        shrink = 3
        all_frames = []

        def synthesize_frame(frame_i, action_i, finger_now, shift):
            this_frame = Image.new("RGB", (dim_h * edge, dim_v * edge))

            move = [0, 0]
            finger_destination = list(finger_now)
            if actions[action_i] == 'left' and finger_now[0]-1 >= 0:
                move[0] = int(-shift*edge)
                finger_destination[0] -= 1
            elif actions[action_i] == 'right' and finger_now[0]+1 < dim_h:
                move[0] = int(shift*edge)
                finger_destination[0] += 1
            elif actions[action_i] == 'up' and finger_now[1]+1 < dim_v:
                move[1] = int(shift*edge)
                finger_destination[1] += 1
            elif actions[action_i] == 'down' and finger_now[1]-1 >= 0:
                move[1] = int(-shift * edge)
                finger_destination[1] -= 1

            # Generate background grid first.
            for i in range(dim_h):
                for j in range(dim_v):
                    this_frame.paste(Image.open(bg[(i+j) % 2]), (i*edge, j*edge, (i+1)*edge, (j+1)*edge))

            # Generate orbs on the background that are not being moved.
            for i in range(dim_h):
                for j in range(dim_v):
                    foreground = Image.open(orbs[boards[frame_i].item((i, j))])
                    if i == finger_now[0] and j == finger_now[1]:
                        pass
                    elif i == finger_destination[0] and j == finger_destination[1]:
                        pass
                    else:
                        this_frame.paste(foreground, (i*edge, j*edge, (i+1)*edge, (j+1)*edge), foreground)

            # Generate the orbs that are being moved.
            i = finger_destination[0]
            j = finger_destination[1]
            foreground = Image.open(orbs[boards[frame_i][i, j]])
            this_frame.paste(foreground, (i*edge-move[0], j*edge-move[1], (i+1)*edge-move[0], (j+1)*edge-move[1]),
                             foreground)
            i = finger_now[0]
            j = finger_now[1]
            foreground = Image.open(orbs[boards[frame_i][i, j]])
            this_frame.paste(foreground, (i*edge+move[0], j*edge+move[1], (i+1)*edge+move[0], (j+1)*edge+move[1]),
                             foreground)
            this_frame  = this_frame.resize((int(dim_h*edge/shrink), int(dim_v*edge/shrink)), Image.ANTIALIAS)
            all_frames.append(this_frame)

        # Generate the whole animation.
        for index in range(len(boards)):
            if index == len(boards) - 1:
                synthesize_frame(index, 0, finger, 0)
            else:
                synthesize_frame(index, index, finger, 0)
                synthesize_frame(index, index, finger, 0.25)
                synthesize_frame(index, index, finger, 0.50)
                synthesize_frame(index, index, finger, 0.75)

                if actions[index] == 'left' and finger[0]-1 >= 0:
                    finger[0] -= 1
                elif actions[index] == 'right' and finger[0]+1 < dim_h:
                    finger[0] += 1
                elif actions[index] == 'up' and finger[1]+1 < dim_v:
                    finger[1] += 1
                elif actions[index] == 'down' and finger[1]-1 >= 0:
                    finger[1] -= 1

        animation = Image.new("RGB", (int(dim_h * edge / shrink), int(dim_v * edge / shrink)))
        animation.save(path, save_all=True, append_images=all_frames, duration=0, loop=100, optimize=True)

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
