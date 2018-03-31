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

import ppad
import ppad.data.dataset01.solved_boards as solved_boards


def smart_data(boards=1, permutations=1, trajectories=1, steps=100, 
               discount=True, gamma=0.9, allowed_orbs=solved_boards.allowed_orbs):
    """
    Generate smart training data in a format that can be directly fed into the learning agent.
    The generation 1 of smart training data is derived from human-solved boards and random sampling.
    :param boards: The number of boards out of the solved boards to randomly use.
    :param permutations: The number of orb identity permutations to perform for each board.
    :param trajectories: The number of trajectories to generate from each permutation of each chosen board.
    :param steps: The number of steps to generate in each trajectory. If steps = -1, terminates the trajectories when
           and only when there is no more combos on the board.
    :param allowed_orbs: A list of allowed orb identities.
    :return: observations, actions and rewards defined exactly as the same-named variables in ppad.pad.game.
    """
    observations = []
    actions = []
    rewards = []
    env = ppad.PAD()

    if boards < 0 or boards > len(solved_boards.boards):
        raise Exception('Invalid input value for board = {0}.'.format(boards))
    if trajectories < 0:
        raise Exception('Invalid input value for traj = {0}.'.format(trajectories))
    if permutations < 0:
        raise Exception('Invalid input value for shuffle = {0}.'.format(permutations))
    if steps < 0:
        raise Exception('Invalid input value for steps = {0}.'.format(steps))

    board_indices = random.sample(range(0, len(solved_boards.boards)), boards)
    for index in board_indices:
        current_board = solved_boards.boards[index]
        for _ in range(permutations):
            # The permutations generated this way are not unique.
            current_permutation = random.sample(allowed_orbs, len(allowed_orbs))
            current_board = permutation_mapping(original_board=current_board,
                                                original_orbs=solved_boards.allowed_orbs,
                                                mapping=current_permutation)
            for _ in range(trajectories):
                env.reset(board=current_board)
                final_reward = env.damage(env.cancel())
                env.reset(board=current_board)
                if steps != -1:
                    for _ in range(steps):
                        action = env.action_space.sample()
                        env.step(action)
                    observations.append(revert_observations(env.observations))
                    actions.append(revert_actions(env.actions))
                    rewards.append(revert_rewards(steps, final_reward))
                elif steps == -1:
                    pass
    
    if discount:
        discounted_rewards_list = []
        for rewards_one_traj in rewards:
            discounted_rewards_list.append(ppad.discount(rewards=rewards_one_traj, gamma=gamma))
        rewards = discounted_rewards_list

    
    return observations, actions, rewards


def permutation_mapping(original_board, original_orbs, mapping):
    """
    Permutate the orb identities based on the input mapping.
    :param original_board: The board to be mapped.
    :param original_orbs: A list of the original orb identities in order.
    :param mapping: A list of the orb identities to be mapped to. For example original_orbs[i] will become mapping[i].
    :return: The board after mapping.
    """
    map = dict()
    for i in range(len(original_orbs)):
        map[original_orbs[i]] = mapping[i]
    mapped_board = np.copy(original_board)
    for index, _ in np.ndenumerate(mapped_board):
        mapped_board[index] = map[mapped_board[index]]
    return mapped_board


def revert_observations(observation):
    return list(reversed(observation))#[:-1]


def revert_actions(actions):
    reversed_actions = list(actions)
    for index in range(len(reversed_actions)):
        action = reversed_actions[index]
        if action == 'left':
            reversed_actions[index] = 'right'
        elif action == 'right':
            reversed_actions[index] = 'left'
        elif action == 'up':
            reversed_actions[index] = 'down'
        elif action == 'down':
            reversed_actions[index] = 'up'
        else:
            raise Exception('revert_actions: invalid action.')
    reversed_actions.append('pass')
    return reversed_actions


def revert_rewards(steps, final_reward):
    rewards = [0] * (steps + 1)
    rewards[-1] = final_reward
    return rewards
