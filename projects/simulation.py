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

#### Imports
import numpy as np
from importlib import reload
import ppad
ppad = reload(ppad)

#### Functions
def model_simulation(model, min_data_points, gamma, 
                     log10_reward = False, beta = None):
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


#### 1. Set-up
## Agent initialization


## Environment and data initialization
env = ppad.PAD()
sar_data = [] #(s,a,r) tuple

## Constants
max_data_size = 10**5
batch_size = 32
log10_reward = True
gamma = 0.99

#### 2. Simulations


































