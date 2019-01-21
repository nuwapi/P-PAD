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
from importlib import reload
import ppad
ppad = reload(ppad)


# FIXME: DEPRECATED!

#### Key functions
def random_trials(episodes, steps, gamma, log10):
    ''' Generates data by taking random actions '''
    observations_list, actions_list, rewards_list = [],[],[]
    for _ in range(episodes):
        env.reset()
    
        for _ in range(steps):
            action = env.action_space.sample()
            env.step(action)
        env.step('pass')

        discounted_rewards = ppad.discount(rewards=env.rewards, gamma=gamma, log10=log10)
        observations_list.append(list(env.observations)) 
        actions_list.append(list(env.actions))
        rewards_list.append(list(discounted_rewards))
    return observations_list, rewards_list, actions_list


#### Agent
agent = ppad.Agent01(learning_rate=0.001, num_filters=128, conv_layers=2)
                #tensorboard_path=os.path.join(os.getcwd(),'data/logs'))



#### 1. Set up
## Observations
observations_list = []
actions_list = []
rewards_list = []
env = ppad.PAD()

## Constants
episodes = 100
steps = 100 # On average it takes 80+ steps to solve the board. 
smart_dims = [10,10,10]
batch_size = 32
log10 = False
gamma = 0.99


#### 2. Sampling
## Random trials
observations, rewards, actions = random_trials(episodes, steps, gamma=gamma, log10=log10)
observations_list.extend(observations)
rewards_list.extend(rewards)
actions_list.extend(actions)

## Just passes
observations, rewards, actions = random_trials(episodes*10, 0, gamma=gamma, log10=log10)
observations_list.extend(observations)
rewards_list.extend(rewards)
actions_list.extend(actions)

## Smart data
observations, actions, rewards = ppad.smart_data(smart_dims[0], smart_dims[1],
                                                 smart_dims[2], gamma=gamma, 
                                                 log10=log10)
observations_list.extend(observations)
rewards_list.extend(rewards)
actions_list.extend(actions)

#### 3. Learning
for iteration in range(1,10):
    # Training
    print('Training...')
    agent.learn(observations=observations_list,
                actions=actions_list,
                rewards=rewards_list,
                iterations=10000,
                experience_replay=True,
                verbose=0)
    
    # Validation - did the net learn anything?
    obs_batch, rew_batch, act_batch = [],[],[]
    idxs = np.random.randint(0,len(observations),batch_size)
    for idx in idxs:
        step = np.random.randint(len(observations[idx]))
        obs_batch.append(observations[idx][step])
        rew_batch.append(rewards[idx][step])
        act_batch.append(actions[idx][step])
    
    x_batch, y_batch = agent.convert_input([obs_batch], [act_batch], [rew_batch])
    print(agent.model.predict(x_batch))


#### 4. Predict and visualize results
for _ in range(100):
    observation = env.reset()
    for _ in range(10):
        action = agent.action(observation)
        observation, _, _, _ = env.step(action=action)
    print(env.actions)
