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
default_team = [{'name': 'Leader 1',
                 'color_1': 'dark',
                 'color_2': 'dark',
                 'atk': 1000,
                 'hp': 2000,
                 'rcv': 100},
                {'name': 'Leader 2',
                 'color_1': 'light',
                 'color_2': 'light',
                 'atk': 1000,
                 'hp': 2000,
                 'rcv': 100},
                 {'name': 'Sub 1',
                 'color_1': 'blue',
                 'color_2': 'green',
                 'atk': 1000,
                 'hp': 2000,
                 'rcv': 100},
                 {'name': 'Sub 2',
                 'color_1': 'green',
                 'color_2': 'red',
                 'atk': 1000,
                 'hp': 2000,
                 'rcv': 100},
                 {'name': 'Sub 3',
                 'color_1': 'red',
                 'color_2': 'blue',
                 'atk': 1000,
                 'hp': 2000,
                 'rcv': 100},
                  {'name': 'Sub 4',
                 'color_1': 'blue',
                 'color_2': None,
                 'atk': 1000,
                 'hp': 2000,
                 'rcv': 100}]


# An array of attributes of one enemy. An array of arrays if there are more than one enemy.
default_enemy = []  # ?

# By default, finger starts at a corner.
default_finger = [0, 0]

# Dictionary used for rendering.
# First element of the tuple is letter representation.
# Second element is color.
default_render_dict = {-1: (' ', ''),
                       0: ('R', '\033[1;31;41m'),
                       1: ('B', '\033[1;34;44m'),
                       2: ('G', '\033[1;32;42m'),
                       3: ('L', '\033[1;33;43m'),
                       4: ('D', '\033[1;35;45m'),
                       5: ('H', '\033[1;30;40m'),
                       6: ('j', '\033[1;30;16m'),
                       7: ('p', '\033[1;5;16m'),
                       8: ('m', '\033[1;90;16m'),
                       9: ('b', '\033[1;94;16m')}


