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
import os
from PIL import Image, ImageDraw, ImageFont

import ppad.pad.parameters as parameters


def cancel(board):
    """
    Cancel all 3+ connected orbs and generate combos.
    :param board: The board for combo cancellation.
    """
    dim_row = board.shape[0]
    dim_col = board.shape[1]

    combos = []
    for i in range(dim_row):
        for j in range(dim_col):
            # If this orb has not been checked.
            orb_type = board[i, j]
            if orb_type != -1:
                # 1. Detect island.
                # Initially, the island only contains the starting orb.
                island = np.zeros((dim_row, dim_col))
                # Detect island starting from position i, j and update island array.
                detect_island(board, island, i, j, orb_type)
                # 2. Prune detected island.
                pruned_island = np.zeros((dim_row, dim_col))
                for k in range(dim_row - parameters.match_n + 1):
                    for l in range(dim_col):
                        if np.sum(island[k:k + 3, l]) == 3:
                            pruned_island[k:k + 3, l] = 1
                for k in range(dim_row):
                    for l in range(dim_col - parameters.match_n + 1):
                        if np.sum(island[k, l:l + 3]) == 3:
                            pruned_island[k, l:l + 3] = 1
                # 3. Save the combo and update board.
                count = np.sum(pruned_island)
                # If there is a combo.
                if count >= parameters.match_n:
                    # Update board.
                    for index, element in np.ndenumerate(pruned_island):
                        if element == 1:
                            board[index] = -1
                    # Generate combo.
                    # TODO: Add shape detection and enhanced orb count in the future.
                    combo = {'color': parameters.int2english[orb_type],
                             'N': count,
                             'enhanced': None,
                             'shape': None}
                    combos.append(combo)
    return combos


def illegal_actions(finger, dim_row=5, dim_col=6):
    illegal_actions = set()
    if finger[0] == 0:
        illegal_actions.add('up')
    elif finger[0] == dim_row - 1:
        illegal_actions.add('down')
    if finger[1] == 0:
        illegal_actions.add('left')
    elif finger[1] == dim_col - 1:
        illegal_actions.add('right')
    return illegal_actions


def detect_island(board, island, x, y, orb_type):
    """
    Recursively detect islands of conneted orbs.
    :param board: A Numpy array of the puzzle board.
    :param island: A Numpy array of the same size as board, initially set to all zeroes.
    :param x: The starting search location.
    :param y: The starting search location.
    :param orb_type: Search for an island of this orb type.
    """
    dim_row = board.shape[0]
    dim_col = board.shape[1]

    if board[x, y] == orb_type:
        island[x, y] = 1
        # Go up.
        if y + 1 < dim_col and island[x, y + 1] == 0:
            detect_island(board, island, x, y + 1, orb_type)
        # Go down.
        if y - 1 >= 0 and island[x, y - 1] == 0:
            detect_island(board, island, x, y - 1, orb_type)
        # Go left.
        if x + 1 < dim_row and island[x + 1, y] == 0:
            detect_island(board, island, x + 1, y, orb_type)
        # Go right.
        if x - 1 >= 0 and island[x - 1, y] == 0:
            detect_island(board, island, x - 1, y, orb_type)


def episode2gif(observations=None, actions=None, filename=None, shrink=3, episode=0, animate=True):
    """
    Note: One needs to have PPADPATH defined to be the root directory (P-PAD) of this repo.
    :param observations: The observations for either a list of episodes or a single episode.
    :param actions: The actions for either a list of episodes or a single episode.
    :param filename: The location where intermediate pngs and the final gif are stored.
    :param shrink: Shrink the output image by this many folds along each dimension.
    :param episode: If observations and actions are list of episodes, specify which episode of those to visualize.
    :param animate: If true, output an animated GIF, it false, output the PNG of the last frame.
    """
    if type(observations) is not list or \
       type(actions) is not list or \
       len(observations) == 0 or \
       len(actions) == 0 or \
       filename is None:
        raise Exception('Invalid input parameters!')
    elif type(observations[0]) is list:
        if len(observations[episode]) < len(actions[episode]):
            raise Exception('len(observations[episode]) should be equal or smaller than len(actions[episode])!')
        single_episode2gif(observations[episode], actions[episode], filename, shrink, animate)
    elif type(observations[0]) is tuple:
        if len(observations) < len(actions):
            raise Exception('len(observations) should be equal or smaller than len(actions)!')
        single_episode2gif(observations, actions, filename, shrink, animate)
    else:
        raise Exception('Elements in list observations do not have the correct type.')


def single_episode2gif(observations, actions, path, shrink=3, animate=True):
    """
    Note: One needs to have PPADPATH defined to be the root directory (P-PAD) of this repo.
    :param observations: The observations of a single episode.
    :param actions: The actions of a single episode.
    :param path: The location where intermediate pngs and the final gif are stored.
    :param shrink: Shrink the output image by this many folds along each dimension.
    :param animate: If true, output an animated GIF, it false, output the PNG of the first frame.
    """
    # Paths to the orb and background assets.
    asset_path = os.environ['PPADPATH'] + '/ppad/assets/'
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
    # Initial finger location of the chosen episode.
    finger = observations[0][1]
    # The size of the board of the chosen episode.
    dim_row = observations[0][0].shape[0]
    dim_col = observations[0][0].shape[1]

    # The length of the square image of the orbs, set to 100px.
    edge = 100
    # Height of the info bar.
    info_bar = 40
    # Font for the texts in the info bar.
    info_font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    # A list to store the per frame images before the final concatenation into a GIF.
    all_frames = []

    def synthesize_frame(frame_i, action_i, finger_now, shift, info_text=''):
        """
        :param frame_i: The current frame to be synthesized.
        :param action_i: The action from the current from to the next frame.
        :param finger_now: The current finger position.
        :param shift: The phase of orb swapping animation, a number from 0 to 1.
        :param info_text: Additional information, combo, step etc.
        """
        this_frame = Image.new('RGB', size=(dim_col * edge, dim_row * edge + info_bar), color=(0, 0, 0, 255))

        move = [0, 0]
        finger_destination = list(finger_now)
        if len(actions) == 0:
            pass
        elif actions[action_i] == 'left' and finger_now[1] - 1 >= 0:
            move[1] = int(-shift * edge)
            finger_destination[1] -= 1
        elif actions[action_i] == 'right' and finger_now[1] + 1 < dim_col:
            move[1] = int(shift * edge)
            finger_destination[1] += 1
        elif actions[action_i] == 'up' and finger_now[0] - 1 >= 0:
            move[0] = int(-shift * edge)
            finger_destination[0] -= 1
        elif actions[action_i] == 'down' and finger_now[0] + 1 < dim_row:
            move[0] = int(shift * edge)
            finger_destination[0] += 1

        # Generate background grid first.
        for i in range(dim_row):
            for j in range(dim_col):
                # Second argument: A 4-tuple defining the left, upper, right, and lower pixel coordinate.
                this_frame.paste(Image.open(bg[(i + j) % 2]), (j*edge, i*edge, (j+1)*edge, (i+1)*edge))

        # Generate orbs on the background that are not being moved.
        for i in range(dim_row):
            for j in range(dim_col):
                foreground = Image.open(orbs[observations[frame_i][0].item((i, j))])
                if i == finger_now[0] and j == finger_now[1]:
                    pass
                elif i == finger_destination[0] and j == finger_destination[1]:
                    pass
                else:
                    this_frame.paste(foreground, (j*edge, i*edge, (j+1)*edge, (i+1)*edge), foreground)

        # Generate the orbs that are being moved.
        i = finger_destination[0]
        j = finger_destination[1]
        foreground = Image.open(orbs[observations[frame_i][0][i, j]])
        this_frame.paste(foreground,
                         (j*edge-move[1], i*edge-move[0], (j+1)*edge-move[1], (i+1)*edge-move[0]),
                         foreground)
        i = finger_now[0]
        j = finger_now[1]
        foreground = Image.open(orbs[observations[frame_i][0][i, j]])
        this_frame.paste(foreground,
                         (j*edge+move[1], i*edge+move[0], (j+1)*edge+move[1], (i+1)*edge+move[0]),
                         foreground)

        draw = ImageDraw.Draw(this_frame)
        draw.text((0, dim_row * edge), info_text, font=info_font, fill=(255, 255, 255, 255))

        this_frame = this_frame.resize((int(dim_col * edge / shrink), int((dim_row * edge + info_bar) / shrink)), Image.ANTIALIAS)
        all_frames.append(this_frame)

    # Generate the whole animation.
    for index in range(len(actions)):
        combos = cancel(np.copy(observations[index][0]))
        display_info = "step = {0}, combo = {1}".format(index+1, len(combos))

        if actions[index] == 'pass':
            synthesize_frame(index, index, finger, 0, display_info)
        else:
            synthesize_frame(index, index, finger, 0, display_info)
            synthesize_frame(index, index, finger, 0.33, display_info)
            synthesize_frame(index, index, finger, 0.67, display_info)

        if actions[index] == 'left' and finger[1] - 1 >= 0:
            finger[1] -= 1
        elif actions[index] == 'right' and finger[1] + 1 < dim_col:
            finger[1] += 1
        elif actions[index] == 'up' and finger[0] - 1 >= 0:
            finger[0] -= 1
        elif actions[index] == 'down' and finger[0] + 1 < dim_row:
            finger[0] += 1

    animation = Image.new('RGB', (int(dim_col * edge / shrink), int((dim_row * edge + info_bar) / shrink)))
    if animate:
        animation.save(path, save_all=True, append_images=all_frames, duration=0, loop=100, optimize=True)
    else:
        all_frames[-1].save(path)
