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

import ppad.learner.somelearner as somelearner
# game should import all other classes needed in ppad.padsimulator
import ppad.padsimilator.game as game

#######################
# Initialization.
similator = game.new()
# The learner should have an API that an take in a generic padsimulator.
# Simulator should have all of the parameters needed for the game itself.
# Learner could have algorithm specific parameters.
# Can take pre-trained policy.
leaner = somelearner.new(similator=similator, policy=None, )

#######################
# Learning.
# learn() should only take in the parameters needed for the learning.
leaner.learn(episodes=1000, epsilon=1e-4)

#######################
# Analysis.
leaner.stat()
leaner.plot()

#######################
# Save results.
pad_policy = leaner.policy()
pad_actionValues = leaner.actionValues()
pad_stateValues = leaner.stateValues()