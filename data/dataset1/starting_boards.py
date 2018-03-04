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

pb1 = np.array([[0, 2, 4, 0, 2],
                [0, 2, 4, 0, 2],
                [0, 2, 4, 0, 2],
                [1, 3, 5, 1, 3],
                [1, 3, 5, 1, 3],
                [1, 3, 5, 1, 3]])

pb2 = np.array([[0, 0, 0, 1, 3],
                [2, 2, 2, 1, 3],
                [3, 3, 3, 1, 3],
                [5, 0, 4, 4, 4],
                [5, 0, 5, 5, 5],
                [5, 0, 2, 2, 2]])

pb3 = np.array([[0, 0, 5, 0, 4],
                [1, 1, 5, 1, 4],
                [2, 2, 5, 2, 4],
                [3, 3, 0, 3, 1],
                [4, 4, 0, 4, 1],
                [5, 5, 0, 5, 1]])

pb4 = np.array([[0, 2, 4, 2, 5],
                [0, 2, 4, 2, 5],
                [0, 2, 4, 2, 5],
                [0, 2, 4, 2, 5],
                [1, 1, 1, 2, 5],
                [3, 3, 3, 0, 5]])

pb5 = np.array([[0, 2, 4, 0, 2],
                [0, 2, 4, 0, 2],
                [0, 2, 4, 0, 2],
                [0, 2, 3, 3, 3],
                [0, 1, 5, 5, 5],
                [1, 1, 1, 1, 1]])


def main():
    board_list = [pb1, pb2, pb3, pb4, pb5]
    for i in range(len(board_list)):
        env = ppad.PAD(board=board_list[i])
        env.episode2gif(path=os.environ['PYTHONPATH']+'/visualizations/perfect_board' + str(i+1) + '.png',
                        shrink=8, ext='png')


if __name__ == "__main__":
    main()
