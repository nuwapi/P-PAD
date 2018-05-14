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

import ppad

import numpy as np

env = ppad.PAD()
agent = ppad.Agent01()

observations_list = []
actions_list = []
rewards_list = []

init_board = np.copy(env.board)

action_sequence = ['left', 'right', 'up', 'down', 'left', 'right', 'up', 'down']
reward = np.array([10, 1, 1, 1, 1, 1, 1, 10])
finger_shift = [0, 0, 0, 0, 1, 1, 1, 1]

# Generate training data.
for i in range(len(action_sequence)):
    init_finger = np.array([2, 2])
    init_finger[0] += finger_shift[i]
    env.reset(board=init_board, finger=init_finger)
    env.step(action_sequence[i])
    env.rewards[0] = reward[i]
    observations_list.append(list(env.observations[:-1]))
    actions_list.append(list(env.actions))
    rewards_list.append(list(env.rewards))

# Generate test data.
# test_board = np.copy(observations_list[0][0][0])
test_board = np.random.randint(6, size=(6, 5))
test_finger = np.copy(observations_list[0][0][1])
test_observations_list = [[(test_board, test_finger)]]
corner = test_observations_list[0][0][0][0, 0]
test_observations_list[0][0][0][0, 0] = 5 - corner
corner = test_observations_list[0][0][0][-1, -1]
test_observations_list[0][0][0][-1, -1] = 5 - corner

# Train and predict.
predict_counter = 0
for _ in range(10000):
    agent.learn(observations=observations_list,
                actions=actions_list,
                rewards=rewards_list,
                iterations=1,
                experience_replay=True,
                verbose=0)

    predict_counter += 1

    # Do prediction once every 20 steps.
    if predict_counter % 20 == 0:
        # [0] here is the x part of the x, y output.
        prediction = agent.model.predict(agent.convert_input(observations=observations_list)[0])
        test_prediction = agent.model.predict(agent.convert_input(observations=test_observations_list)[0])

        # Take the first four actions of the first frame and the first four actions of the second frame.
        # Note that frames 1, 2, 3 are the same as 0 and 5, 6, 7 are the same as 4.
        diff = np.concatenate((prediction[0][0:4], prediction[4][0:4]), axis=0) - reward
        # There shouldn't be a big difference between the two boards with only 2 corner swapped.
        test_diff = test_prediction[0][0] - reward[0]

        print('Absolute error for the training set:')
        print(diff)
        print('Difference for the test data point:')
        print(test_prediction[0][0], reward[0])

        predict_counter = 0
