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
import ppad
ppad = reload(ppad)
from ppad.agent.agent01 import Agent01

sess = tf.Session()
agent = Agent01(sess)

print('\nTraining step 1.')
batch_size = 64
boards = np.ones((batch_size, 5, 6), dtype=np.int32)
fingers = np.zeros((batch_size, 2), dtype=np.int32)
targets = np.random.rand(batch_size, 5)
agent.train(boards, fingers, targets)

print('\nTraining step 2.')
batch_size = 32
boards = np.ones((batch_size, 5, 6), dtype=np.int32)
fingers = np.zeros((batch_size, 2), dtype=np.int32)
targets = np.random.rand(batch_size, 5)
agent.train(boards, fingers, targets)

print('\nDo a predict with A.')
board = np.ones((1, 5, 6), dtype=np.int32)
finger = np.zeros((1, 2), dtype=np.int32)
print('Rewards:', agent.predict(board, finger, model='A'))

print('\nDo an act with A.')
board = np.ones((1, 5, 6), dtype=np.int32)
finger = np.zeros((1, 2), dtype=np.int32)
print('Model chooses:', agent.act(board, finger, model='A'))

print('\nDo a predict with B.')
board = np.ones((1, 5, 6), dtype=np.int32)
finger = np.zeros((1, 2), dtype=np.int32)
print('Rewards:', agent.predict(board, finger, model='B'))

print('\nDo an act with B.')
board = np.ones((1, 5, 6), dtype=np.int32)
finger = np.zeros((1, 2), dtype=np.int32)
print('Model chooses:', agent.act(board, finger, model='B'))

print('\nCopy A to B.')
agent.copy_A_to_B(verbose=True)

print('\nDo a predict with B.')
board = np.ones((1, 5, 6), dtype=np.int32)
finger = np.zeros((1, 2), dtype=np.int32)
print('Rewards:', agent.predict(board, finger, model='B'))

print('\nDo an act with B.')
board = np.ones((1, 5, 6), dtype=np.int32)
finger = np.zeros((1, 2), dtype=np.int32)
print('Model chooses:', agent.act(board, finger, model='B'))