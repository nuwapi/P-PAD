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

import os
from importlib import reload
import numpy as np
import random
import tensorflow as tf
from random import shuffle, sample
import ppad
ppad = reload(ppad)
from ppad.agent.agent01 import Agent01
from ppad.pad.utils import illegal_actions


############################
# 0. Define functions.
############################
def model_simulation(agent, min_data_points, gamma, log10_reward=False,
                     policy='max', beta=None, max_episode_len=200,
                     visualize=False):
    """
    Generates data step by step according to model policy.
    """
    sim_sars = []
    num_episodes = 0
    
    while len(sim_sars) < min_data_points:
        env.reset()

        counter = 0
        action = None
        num_episodes += 1
        
        while action != 'pass':
            if counter > max_episode_len:
                action = 'pass'
                agent.last_action = ACTION2ID['pass']
            else:
                action = agent.act(env.board, env.finger, 'A', method=policy, beta=beta)
            env.step(action)
            counter += 1
        
        discounted_rewards = ppad.discount(rewards=env.rewards, gamma=gamma, log10=log10_reward)
        sim_sars.extend(zip(list(env.observations), list(env.actions), list(discounted_rewards)))
        
        if visualize:
            env.visualize(filename='/home/chris/Documents/model_trajectory.gif')
            print('* Saving the model trajectory.')
            break

    return sim_sars, num_episodes


def model_simulation_2sla(agent, min_data_points, gamma, log10_reward=False,
                          policy='max', beta=None, max_episode_len=50,
                          max_lookahead_len=10, visualize=False):
    """
    Generates data step by step according to model policy.
    The 2-step look ahead (2sla) version.
    """
    sim_sars = []
    num_episodes = 0

    # We want to collect at least min_data_points number of data points.
    while len(sim_sars) < min_data_points:
        # Reset the environment to start a new simulation.
        env.reset()
        num_episodes += 1

        # Simulate and finish an episode before going into 2sla.
        action = None
        last_action = 'pass'
        episode_len_counter = 0
        while action != 'pass':
            last_action = ID2ACTION[agent.last_action]
            if episode_len_counter >= max_episode_len:
                action = 'pass'
            else:
                action = agent.act(env.board, env.finger, 'A', method=policy, beta=beta)
                episode_len_counter += 1

            if action != 'pass':
                env.step(action)

        # Do 2-step look ahead off-policy when the episode finishes naturally.
        all_action_sequences = []
        all_rewards = []

        # Priority 1: The original pass.
        all_action_sequences.append(['pass'])
        all_rewards.append(env.calculate_reward(board=env.board, skyfall_damage=False))

        # Priority 2: 1-step look ahead.
        # FIXME: What happens if action makes env go out of bounds?
        invalid_actions = illegal_actions(env.finger)
        invalid_actions.add(env.action_space.opposite_actions[last_action])
        step1_actions = list(NON_PASS_ACTIONS - invalid_actions)
        random.shuffle(step1_actions)
        for action in step1_actions:
            env.step(action)
            all_action_sequences.append([action, 'pass'])
            all_rewards.append(env.calculate_reward(board=env.board, skyfall_damage=False))
            env.backtrack(n=1)

        # Priority 3: 2-step look ahead plus free roam.
        random.shuffle(step1_actions)
        for step1_action in step1_actions:
            env.step(step1_action)

            invalid_actions = illegal_actions(env.finger)
            invalid_actions.add(env.action_space.opposite_actions[step1_action])
            step2_actions = list(NON_PASS_ACTIONS - invalid_actions)
            random.shuffle(step2_actions)

            for step2_action in step2_actions:
                steps_made = 1
                env.step(step2_action)
                agent.last_action = ACTION2ID[step2_action]
                roaming_action = ''
                current_action_list = [step1_action, step2_action]

                while roaming_action != 'pass':
                    if steps_made >= max_lookahead_len:
                        roaming_action = 'pass'
                        agent.last_action = ACTION2ID['pass']
                    else:
                        roaming_action = agent.act(env.board, env.finger, 'A', method=policy, beta=beta)
                    current_action_list.append(roaming_action)

                    if roaming_action != 'pass':
                        env.step(roaming_action)
                        steps_made += 1
                    else:
                        all_rewards.append(env.calculate_reward(board=env.board, skyfall_damage=False))
                        all_action_sequences.append(current_action_list)

                        # Backtrack step 2s.
                        env.backtrack(n=steps_made)
                        break

            # Backtrack step 1.
            env.backtrack(n=1)

        max_reward = max(all_rewards)
        max_reward_index = 0
        assert len(all_rewards) == len(all_action_sequences)
        # The appending order ensures priority.
        for i in range(len(all_rewards)):
            if max_reward == all_rewards[i]:
                max_reward_index = i

        assert agent.last_action == ACTION2ID['pass']
        for action in all_action_sequences[max_reward_index]:
            # TODO: Check that pass has been been passed to env prior to this.
            # TODO: Also, what is the behavior of env when pass is given?
            env.step(action)

        discounted_rewards = ppad.discount(rewards=env.rewards, gamma=gamma, log10=log10_reward)
        sim_sars.extend(zip(list(env.observations), list(env.actions), list(discounted_rewards)))

        if visualize:
            env.visualize(filename='/home/chris/Documents/model_trajectory.gif')
            print('* Saving the model trajectory.')
            break

    return sim_sars, num_episodes


############################
# 1. Define constants.
############################

# The number of steps of the simulation/training.
STEPS = 10**5
# The max experience replay buffer size. This is a hyperparameter, people usually use 1,000,000.
MAX_DATA_SIZE = 10**5
# Training batch size.
BATCH_SIZE = 32
# Update model B every this number of steps.
B_UPDATE_FREQ = 10
# If a game episode doesn't end after this number of steps, give 'pass' to the env.
MAX_EPISODE_LEN = 50
# The minimum number of SAR pairs to collect per training step.
MIN_STEP_SARS = 200

# Take log10 of the raw reward value or not.
LOG10_REWARD = False
# Discount rate gamma.
GAMMA = 0.95

# Exploration policy.
POLICY = 'boltzmann'
# The initial and max beta for Boltzmann policy

if LOG10_REWARD:
    BETA_INIT = 10**-2
    MAX_BETA = 5
else:
    BETA_INIT = 10**-4
    MAX_BETA = 10**-2
# Beta increases every this number of steps.
BETA_INCREASE_FREQ = 200  
# The ratio of beta increase.
BETA_INCREASE_RATE = 1.25

# Look ahead length.
MAX_LOOKAHEAD_LEN = 10

# Save every this number of steps.
SAVE_FREQ = 10**3
# Save path.
SAVE_PATH = './model'
# Report metrics every this number of steps.
REPORT_FREQ = 10

# Action dictionary.
ACTION2ID = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'pass': 4}
ID2ACTION = {0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'pass'}
NON_PASS_ACTIONS = {'up', 'down', 'left', 'right'}


############################
# 2. Set-up.
############################

# Agent initialization.
sess = tf.Session()
agent = Agent01(sess, conv_layers=((2, 128), (2, 64)),
                dense_layers=(64, 32, 5), learning_rate=0.0001)
agent.copy_A_to_B()

# Environment initialization.
env = ppad.PAD(skyfall_damage=False)

# (s,a,r) tuples.
sar_data = []

# Metrics variables.
beta = BETA_INIT
print('BETA value at the end of training:', BETA_INIT*BETA_INCREASE_RATE**(STEPS/BETA_INCREASE_FREQ))
total_loss = 0
total_rmse = 0
total_reward = 0
total_new_data_points = 0
total_actions = 0
total_episodes = 0
max_reward = 0


############################
# 3. Simulation.
############################

for step in range(STEPS):
    # a. Update beta if needed.
    if POLICY == 'boltzmann' and ((step + 1) % BETA_INCREASE_FREQ == 0):
        beta *= BETA_INCREASE_RATE
        if beta>MAX_BETA:
            beta = MAX_BETA
        print('* Beta updated to {0}'.format(beta))

    # b. Generate new training data.
    # sar_new, num_episodes = model_simulation(agent, min_data_points=MIN_STEP_SARS, gamma=GAMMA,
    #                                          log10_reward=LOG10_REWARD, policy=POLICY, beta=beta,
    #                                          max_episode_len=MAX_EPISODE_LEN)
    sar_new, num_episodes = model_simulation_2sla(agent, min_data_points=MIN_STEP_SARS, gamma=GAMMA,
                                             log10_reward=LOG10_REWARD, policy=POLICY, beta=beta,
                                             max_episode_len=MAX_EPISODE_LEN, max_lookahead_len=MAX_LOOKAHEAD_LEN)

    new_data_len = len(sar_new)
    total_new_data_points += new_data_len
    total_episodes += num_episodes
    
    # c. Combine new training data with the current list.
    sar_data += sar_new
    # Discard the extra data.
    if len(sar_data) > MAX_DATA_SIZE:
        # TODO: Inefficient.
        shuffle(sar_data)
        sar_data = sar_data[0:MAX_DATA_SIZE]

    # d. Do training.
    sar_train = sar_new + [sar_data[i] for i in sample(range(len(sar_data)), len(sar_new))]
    shuffle(sar_train)
    no_of_mini_batches = int(len(sar_train) / BATCH_SIZE)
    for training_step in range(no_of_mini_batches):
        batch_data = sar_train[training_step * BATCH_SIZE:(training_step + 1) * BATCH_SIZE]

        # Turn states into np format.
        states, actions, rewards = map(list, zip(*batch_data))
        boards, fingers = map(list, zip(*states))
        actions = [ACTION2ID[action] for action in actions]
        boards = np.array(boards)
        fingers = np.array(fingers)

        # Use model B to do a prediction for actions with unknown rewards.
        targets = agent.predict(boards, fingers, 'B')
        for i, action in enumerate(actions):
            targets[i, action] = rewards[i]

        # Train A.
        loss, q_values = agent.train(boards, fingers, targets)
        loss = np.sqrt(loss)

        # Calculate root mean sqaure error.
        rmse = 0
        for i, action in enumerate(actions):
            rmse += (targets[i, action] - q_values[i, action])**2
        rmse = np.sqrt(rmse/len(actions))

        total_rmse += rmse / no_of_mini_batches
        total_loss += loss / no_of_mini_batches
        total_reward += sum(rewards) / BATCH_SIZE / no_of_mini_batches
        max_reward = max(max_reward, max(rewards))

    # e. Do report.
    if (step + 1) % REPORT_FREQ == 0:
        print('============================> STEP {0} OUT OF {1}.'.format(step + 1, STEPS))
        print('New SAR pairs  = {0}.'.format(total_new_data_points))
        print('Avg len per ep = {:.4f}.'.format(total_new_data_points / total_episodes))
        print('Avg RMS error  = {:.4f}.'.format(total_rmse / REPORT_FREQ))
        print('Avg loss^0.5   = {:.4f}.'.format(total_loss / REPORT_FREQ))
        print('Avg reward     = {:.4f}.'.format(total_reward / REPORT_FREQ))
        print('Max reward     = {:.4f}.'.format(max_reward))
        total_new_data_points = 0
        total_episodes = 0
        total_loss = 0
        total_rmse = 0
        total_reward = 0
        max_reward = 0

    # f. Update model B.
    if (step + 1) % B_UPDATE_FREQ == 0:
        print('* Updating model B with model A.')
        agent.copy_A_to_B()

    # g. Save model.
    if (step + 1) % SAVE_FREQ == 0:
        print('* Saving the model.')
        checkpoint_name = 'model-ckpt.' + str(step)
        checkpoint_name = os.path.join(SAVE_PATH, checkpoint_name)
        agent.save(checkpoint=checkpoint_name)
