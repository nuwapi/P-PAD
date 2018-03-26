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
    :param log10: Take log based 10 of rewards.
    :return: Discounted rewards.
    """
    discounted_rewards = np.zeros_like(rewards)
    current_reward = 0.0

    for i in reversed(range(0, len(rewards))):
        current_reward = current_reward * gamma + rewards[i]
        discounted_rewards[i] = current_reward

    # If we want to normalize the rewards.
    if norm:
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards)
        if std > 0:
            discounted_rewards = (discounted_rewards - mean) / std
    
    # Or, if we want to take the log of rewards
    if log10:
        discounted_rewards = np.log10(discounted_rewards)

    return discounted_rewards


def step_penalty(rewards, penalty, norm=True):
    pass
