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
import os

import ppad

# 16 solved boards.
solved_boards = [np.array([[5, 5, 5, 5, 3],  # 1_3_5_6_7_8
                           [0, 0, 0, 0, 3],
                           [1, 1, 5, 1, 3],
                           [0, 0, 5, 0, 3],
                           [2, 2, 5, 2, 3],
                           [1, 1, 5, 1, 4]]),
                 np.array([[5, 3, 5, 5, 5],  # 1_4_5_5_6_9
                           [5, 3, 1, 2, 0],
                           [5, 3, 1, 2, 0],
                           [3, 5, 1, 2, 0],
                           [3, 5, 1, 2, 0],
                           [3, 5, 1, 2, 4]]),
                 np.array([[0, 0, 0, 3, 1],  # 1_4_5_6_7_7
                           [1, 4, 2, 3, 1],
                           [1, 4, 0, 3, 1],
                           [1, 4, 0, 3, 4],
                           [1, 4, 0, 3, 4],
                           [5, 5, 5, 5, 4]]),
                 np.array([[4, 5, 2, 2, 2],  # 1_5_5_6_6_7
                           [4, 2, 0, 0, 0],
                           [4, 2, 4, 4, 4],
                           [4, 2, 0, 0, 0],
                           [3, 3, 3, 3, 3],
                           [1, 1, 1, 1, 1]]),
                 np.array([[4, 4, 4, 4, 4],  # 2_2_4_6_8_8
                           [1, 1, 1, 1, 1],
                           [0, 0, 2, 5, 4],
                           [3, 3, 3, 5, 4],
                           [1, 1, 1, 5, 4],
                           [5, 5, 5, 2, 4]]),
                 np.array([[0, 3, 4, 4, 4],  # 2_3_4_5_6_10
                           [0, 3, 0, 5, 4],
                           [0, 3, 0, 5, 4],
                           [4, 1, 0, 5, 1],
                           [4, 2, 2, 2, 2],
                           [4, 0, 0, 0, 0]]),
                 np.array([[0, 4, 2, 2, 2],  # 2_4_4_6_7_7
                           [0, 5, 5, 5, 5],
                           [0, 4, 3, 0, 1],
                           [5, 2, 3, 0, 1],
                           [5, 2, 3, 0, 1],
                           [5, 2, 3, 0, 1]]),
                 np.array([[4, 4, 4, 4, 4],  # 2_4_5_5_7_7
                           [2, 2, 2, 2, 2],
                           [5, 5, 1, 0, 3],
                           [1, 3, 1, 0, 3],
                           [1, 3, 1, 0, 3],
                           [1, 3, 1, 0, 3]]),
                 np.array([[3, 4, 0, 4, 5],  # 3_3_4_5_6_9
                           [3, 4, 0, 4, 5],
                           [3, 4, 0, 4, 5],
                           [4, 2, 0, 2, 2],
                           [4, 5, 0, 5, 5],
                           [4, 1, 1, 1, 1]]),
                 np.array([[1, 3, 4, 0, 5],  # 3_3_4_6_6_8
                           [1, 3, 4, 0, 5],
                           [1, 3, 4, 0, 5],
                           [1, 5, 5, 5, 1],
                           [1, 3, 3, 3, 1],
                           [2, 2, 2, 2, 1]]),
                 np.array([[2, 1, 0, 3, 1],  # 3_3_5_5_7_7
                           [2, 1, 0, 3, 1],
                           [2, 1, 0, 3, 1],
                           [2, 1, 2, 2, 2],
                           [4, 4, 4, 4, 4],
                           [5, 5, 5, 5, 5]]),
                 np.array([[1, 5, 5, 5, 0],  # 3_4_4_5_6_8
                           [1, 4, 2, 5, 0],
                           [1, 4, 2, 5, 0],
                           [1, 4, 2, 0, 3],
                           [1, 4, 2, 0, 3],
                           [5, 5, 5, 0, 3]]),
                 np.array([[4, 4, 4, 4, 4],  # 3_4_5_5_6_7
                           [5, 1, 1, 1, 1],
                           [5, 2, 2, 2, 2],
                           [5, 0, 1, 3, 0],
                           [5, 0, 1, 3, 0],
                           [5, 0, 1, 3, 0]]),
                 np.array([[5, 5, 3, 5, 5],  # 4_4_4_5_6_7
                           [1, 1, 3, 1, 1],
                           [2, 2, 3, 2, 2],
                           [0, 0, 3, 0, 4],
                           [4, 4, 3, 4, 4],
                           [0, 0, 0, 0, 4]]),
                 np.array([[3, 5, 5, 5, 2],  # 4_5_5_5_5_6
                           [3, 2, 1, 5, 2],
                           [3, 2, 1, 5, 2],
                           [3, 2, 1, 1, 1],
                           [0, 0, 0, 0, 0],
                           [4, 4, 4, 4, 4]]),
                 np.array([[0, 0, 0, 0, 0],  # 5_5_5_5_5_5
                           [1, 1, 1, 1, 1],
                           [2, 2, 2, 2, 2],
                           [3, 3, 3, 3, 3],
                           [5, 5, 5, 5, 5],
                           [4, 4, 4, 4, 4]])]


def main():
    visualize_n_boards = 5
    for i in range(visualize_n_boards):
        env = ppad.PAD(board=solved_boards[i])
        env.episode2gif(path=os.environ['PYTHONPATH']+'/visualizations/solved_board' + str(i+1) + '.png',
                        shrink=8, ext='png')


if __name__ == "__main__":
    main()
