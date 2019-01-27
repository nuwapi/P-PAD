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
def model_simulation(agent, min_data_points, gamma, log10_reward = False, 
                     policy = 'max', beta = None, max_episode_len = 200):
    ''' Generates data step by step according to model policy '''
    sim_sars = []
    while len(sim_sars)<min_data_points:
        env.reset()
        
        counter = 0
        action = None
        while action != 'pass':
            if counter > max_episode_len:
                action = 'pass'
            else:
                action = agent.act(env.board, env.finger, 'A', method=policy, beta=beta)
            env.step(action)
            counter +=1

        
        discounted_rewards = ppad.discount(rewards=env.rewards, gamma=gamma, log10=log10_reward)
        sim_sars.extend(zip(list(env.observations), 
                            list(env.actions), 
                            list(discounted_rewards)))
    return sim_sars


############################
# 1. Set-up.
############################

# Constants.
# The max experience replay buffer size. This is a hyperparameter, people usually use 1,000,000.
MAX_DATA_SIZE = 10**5
# Training batch size.
BATCH_SIZE = 32
# The number of steps of the simulation/training.
STEPS = 10**5
STEP_REPORT_FREQ = 10
# Update model B every this number of steps.
B_UPDATE_FREQ = 10
# Number of episodes to generate per data update.
EPISODES_PER_STEP = 15
# If a game episode doesn't end after this number of steps, give 'pass' to the env.
MAX_EPISODE_LEN = 200
MIN_STEP_SARS = 200
LOG10_REWARD = True
GAMMA = 0.99
# Action dictionary.
ACTION2ID = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'pass': 4}
# Exploration policy
POLICY = 'boltzmann'
BETA_INIT = 0.1
BETA_RATE_INCREASE = 1.1 # Magnitude of increase in beta when changed
BETA_INCREASE_FREQUENCY = 100 # Number of episodes before decay 
print(BETA_INIT*BETA_RATE_INCREASE**(STEPS/BETA_INCREASE_FREQUENCY))

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

beta = BETA_INIT
new_data_points = 0
for step in range(STEPS):
    
    if POLICY=='boltzmann' and ((step+1) % BETA_INCREASE_FREQUENCY == 0):
        beta *= BETA_RATE_INCREASE
        print('* Beta updated to {0}'.format(beta))

    # a. Generate training data.
    sar_new = model_simulation(agent, MIN_STEP_SARS, GAMMA, log10_reward = LOG10_REWARD, 
                               policy = POLICY, beta = beta, max_episode_len = MAX_EPISODE_LEN)
    new_data_len = len(sar_new)
    new_data_points += new_data_len

    # b. Combine new training data with the current list.
    sar_data += sar_new
    # Discard the extra data.
    if len(sar_data) > MAX_DATA_SIZE:
        shuffle(sar_data) # TODO: Inefficient. 
        sar_data = sar_data[0:MAX_DATA_SIZE]

    # c. Do training.
    no_of_mini_batches = int(new_data_len/BATCH_SIZE) + 1
    total_loss = 0
    total_reward = 0
    for training_step in range(no_of_mini_batches):
        batch_data_new = sar_new[training_step*BATCH_SIZE:(training_step+1)*BATCH_SIZE]
        # Do a random sample.
        batch_data_replay = [sar_data[i] for i in sample(range(len(sar_data)), BATCH_SIZE)]

        for batch_data in [batch_data_new, batch_data_replay]:
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

    # Communicate the current status
    if ((step % STEP_REPORT_FREQ) == 0):
        print('============================> STEP {0} OUT OF {1}.'.format(step + 1, STEPS))
        print('New SAR pairs generated: {0}.'.format(new_data_points))
        print('Avg loss   = {0}.'.format(total_loss/no_of_mini_batches))
        print('Avg reward = {0}.'.format(total_reward/no_of_mini_batches/BATCH_SIZE))
        new_data_points = 0

    # d. Update B.
    if (step + 1) % B_UPDATE_FREQ == 0:
        print('* Updating model B with model A.')
        agent.copy_A_to_B()




















