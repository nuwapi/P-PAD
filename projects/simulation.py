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

from importlib import reload
import numpy as np
import tensorflow as tf
from random import shuffle, sample
import ppad
ppad = reload(ppad)
from ppad.agent.agent01 import Agent01


#### Functions
def model_simulation(model, min_data_points, gamma, log10_reward = False, beta = None):
    ''' Generates data step by step accorind got model policy '''
    sim_sars = []
    while len(sim_sars)<min_data_points:
        env.reset()
        
        action = None
        while action != 'pass':
            action = model.act(env.board)
            env.step(action)
        
        discounted_rewards = ppad.discount(rewards=env.rewards, gamma=gamma, log10=log10_reward)
        sim_sars.extend(zip(list(env.observations), 
                            list(env.actions)), 
                            list(discounted_rewards))
    return sim_sars


############################
# 1. Set-up.
############################

# Constants.
# The max experience replay buffer size. This is a hyperparameter, people usually use 1,000,000.
MAX_DATA_SIZE = 10**5
# Training batch size.
BATCH_SIZE = 10
# The number of steps of the simulation/training.
STEPS = 10**4
# Update model B every this number of steps.
B_UPDATE_FREQ = 32
# Number of episodes to generate per data update.
EPISODES_PER_STEP = 10
# If a game episode doesn't end after this number of steps, give 'pass' to the env.
MAX_EPISODE_LEN = 150
LOG10_REWARD = True
GAMMA = 0.99
# Action dictionary.
ACTION2ID = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'pass': 4}

# Agent initialization.
sess = tf.Session()
agent = Agent01(sess)
agent.copy_A_to_B()

# Environment initialization.
env = ppad.PAD()

# (s,a,r) tuples.
sar_data = []

############################
# 2. Simulation.
############################

for step in range(STEPS):
    print('============================> STEP {0} OUT OF {1}.'.format(step + 1, STEPS))

    # a. Generate training data.
    sar_new = []
    for episode in range(EPISODES_PER_STEP):
        counter = 1
        env.reset()
        action = ''
        while action != 'pass' and counter < MAX_EPISODE_LEN:
            action = agent.act(env.board, env.finger, 'A')
            env.step(action)
            counter += 1
        # If the episode went beyond MAX_EPISODE_LEN, end the episode.
        if action != 'pass':
            env.step('pass')

        discounted_rewards = ppad.discount(rewards=env.rewards, gamma=GAMMA, log10=LOG10_REWARD)
        for s, a, r in zip(env.observations, env.actions, discounted_rewards):
            sar_new.append((s, a, r))

    new_data_len = len(sar_new)
    print('Generated {0} new SAR pairs.'.format(new_data_len))

    # b. Combine new training data with the current list.
    print('Updating sar_data.')
    sar_data += sar_new
    # Discard the extra data.
    if len(sar_data) > MAX_DATA_SIZE:
        shuffle(sar_data)
        sar_data = sar_data[0:MAX_DATA_SIZE]

    # c. Do training.
    print('Doing training.')
    no_of_mini_batches = int(new_data_len/BATCH_SIZE) + 1
    total_loss = 0
    total_reward = 0
    for training_step in range(no_of_mini_batches):
        # Do a random sample.
        batch_data = [sar_data[i] for i in sample(range(len(sar_data)), BATCH_SIZE)]

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
        loss = agent.train(boards, fingers, targets)
        total_loss += loss
        total_reward += sum(rewards)

    print('Avg loss   = {0}.'.format(total_loss/no_of_mini_batches))
    print('Avg reward = {0}.'.format(total_reward/no_of_mini_batches/BATCH_SIZE))

    # d. Update B.
    if (step + 1) % B_UPDATE_FREQ == 0:
        print('* Updating model B with model A.')
        agent.copy_A_to_B()




















