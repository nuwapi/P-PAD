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

# The sum of all 10 skyfall rates equals to 1.
# In order, the 10 orb types are:
#   [red,
#    blue,
#    green,
#    light,
#    dark,
#    heal,
#    jammer,
#    poison,
#    mortal poison,
#    bomb]
default_skyfall = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0, 0, 0, 0]

# By default, there is no locked orbs. The orb identities in order are the same as 'skyfall'.
# The locked probability for individual orb types is between 0 to 1.
default_skyfall_locked = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# By default, there is no enhanced orbs. The orb identities in order are the same as 'skyfall'.
# The enhanced probability for individual orb types is between 0 to 1.
default_skyfall_enhanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# An array of 6 arrays. The first two are leaders.
default_team = [[], []]  # ?

# An array of attributes of one enemy. An array of arrays if there are more than one enemy.
default_enemy = []  # ?

# By default, finger starts at a corner.
default_finger = [0, 0]

