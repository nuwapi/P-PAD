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

import keras
from keras.layers import Conv2D, Dense, Flatten
import numpy as np


class Agent01:
    def __init__(self):
        # Basic parameters.
        num_filters = 8
        kernel_len = 3
        input_shape = (6, 5, 7)  # 6 by 5 board with 7 channels (5 colors plus heal and finger position).
        dense_units = 32
        num_classes = 5  # left, right, up, down and pass.

        # Initialize model.
        self.model = keras.models.Sequential()
        self.model.add(Conv2D(filters=num_filters,
                              kernel_size=(kernel_len, kernel_len),
                              activation='relu',
                              input_shape=input_shape,
                              kernel_initializer=keras.initializers.RandomNormal()))
        self.model.add(Conv2D(filters=num_filters,
                              kernel_size=(kernel_len, kernel_len),
                              activation='relu',
                              kernel_initializer=keras.initializers.RandomNormal()))
        self.model.add(Flatten())
        self.model.add(Dense(units=dense_units,
                             activation='relu',
                             kernel_initializer=keras.initializers.RandomNormal()))
        self.model.add(Dense(units=num_classes,
                             activation='softmax',
                             kernel_initializer=keras.initializers.RandomNormal()))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])

    def learn(self, observations, actions, rewards, epochs, batch_size=1):
        # Numpy array for all samples.
        x, y = self.convert_input(observations, actions, rewards)
        # Numpy array for one sample.
        x1 = np.zeros((1, 6, 5, 7))
        y1 = np.zeros((1, 5))

        # For each state-action pair.
        for i in range(len(x)):
            x1[0] = x[i]
            y1[0] = y[i]
            yp = self.model.predict(x1)
            # For the actions that were not chosen, assign their reward value to what the neural net predicts.
            # Effectively, the neural net doesn't learn from un-chosen actions.
            y1[y1 == np.nan] = yp[y1 == np.nan]

            self.model.fit(x=x1,
                           y=y1,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1)

    def action(self, observation):
        observations = [[observation]]
        prediction = self.model.predict(self.convert_input(observations=observations)[0])
        action_type = np.unravel_index(np.argmax(prediction[0], axis=None), prediction[0].shape)
        action_type = int(action_type[0])
        if action_type == 0:
            return 'left'
        elif action_type == 1:
            return 'right'
        elif action_type == 2:
            return 'up'
        elif action_type == 3:
            return 'down'
        elif action_type == 4:
            return 'pass'
        else:
            raise Exception('Action type {0} is invalid.'.format(action_type))

    @staticmethod
    def convert_input(observations=None, actions=None, rewards=None):
        """
        Convert the output of the environment into Tensorflow usable input.
        :param observations: observations[i][j] = (np_array(6,5), np_array(2))
        :param actions: actions[i][j] = string
        :param rewards: rewards[i][j] = float
        :return:
        """
        x = None
        y = None

        if observations is not None:
            # Assuming all episodes have the same number of steps.
            x = np.zeros((len(observations) * len(observations[0]), 6, 5, 7))
            counter = 0
            # Looping through episodes.
            for i in range(len(observations)):
                # Looping through steps in episodes.
                for j in range(len(observations[0])):
                    board = observations[i][j][0]
                    finger = observations[i][j][1]
                    x[counter][:, :, 0][board == 0] = np.ones((6, 5))[board == 0]
                    x[counter][:, :, 1][board == 1] = np.ones((6, 5))[board == 1]
                    x[counter][:, :, 2][board == 2] = np.ones((6, 5))[board == 2]
                    x[counter][:, :, 3][board == 3] = np.ones((6, 5))[board == 3]
                    x[counter][:, :, 4][board == 4] = np.ones((6, 5))[board == 4]
                    x[counter][:, :, 5][board == 5] = np.ones((6, 5))[board == 5]
                    x[counter][finger[0], finger[1], 6] = 1
                    counter += 1

        if actions is not None and rewards is not None:
            # Assuming all episodes have the same number of steps.
            y = np.zeros((len(actions) * len(actions[0]), 5))
            counter = 0
            # Looping through episodes.
            for i in range(len(actions)):
                # Looping through steps in episodes.
                for j in range(len(actions[0])):
                    action = actions[i][j]
                    if action == 'left':
                        y[counter][0] = rewards[i][j]
                    elif action == 'right':
                        y[counter][1] = rewards[i][j]
                    elif action == 'up':
                        y[counter][2] = rewards[i][j]
                    elif action == 'down':
                        y[counter][3] = rewards[i][j]
                    elif action == 'pass':
                        y[counter][4] = rewards[i][j]
                    else:
                        raise Exception('Action {0} is invalid.'.format(action))
                    counter += 1

        return x, y


def agent01():
    return Agent01()
