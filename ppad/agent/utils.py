"""
P-PAD
Copyright (C) 2018 NWP, CP
s
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


def discount(rewards, gamma, norm=False, log10=True):
    """
    Discount rewards.
    :param rewards: Raw rewards.
    :param gamma: Discount rate.
    :param norm: Normalize the discounted rewards or not.
    :param log10: Take log base 10 of rewards.
    :return: Discounted rewards.
    """
    discounted_rewards = np.array(rewards)
    current_reward = 0.0
    
    if any(discounted_rewards>0):
        # Damage is often in the 1000s. Take log10?
        if log10:
            discounted_rewards[discounted_rewards>0] = np.log10(discounted_rewards[discounted_rewards>0])
        
        # Discount the rewards.
        for i in reversed(range(0, len(discounted_rewards))):
            current_reward = current_reward * gamma + discounted_rewards[i]
            discounted_rewards[i] = current_reward
    
        # Normalize the rewards?
        if norm:
            mean = np.mean(discounted_rewards)
            std = np.std(discounted_rewards)
            if std > 0:
                discounted_rewards = (discounted_rewards - mean) / std

    return discounted_rewards


def step_penalty(rewards, penalty, norm=True):
    pass
