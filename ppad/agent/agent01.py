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
    def __init__(self, num_filters=32, kernel_len=3, dense_units=64,
                 learning_rate = 0.01, conv_layers=2, dense_layers=2,
                 tensorboard_path='/data/logs'):
        # Basic parameters.
        input_shape = (6, 5, 7)  # 6 by 5 board with 7 channels (5 colors plus heal and finger position).
        num_classes = 5  # left, right, up, down and pass.
        if type(kernel_len) is int:
            kernel_len = [kernel_len]*conv_layers
        
        # Initialize model.
        self.model = keras.models.Sequential()
        for i in range(conv_layers):
            self.model.add(Conv2D(filters=num_filters,
                                  kernel_size=(kernel_len[i], kernel_len[i]),
                                  activation='relu',
                                  input_shape=input_shape,
                                  kernel_initializer=keras.initializers.RandomNormal()))
        self.model.add(Flatten())
        
        for i in range(dense_layers):
            self.model.add(Dense(units=dense_units,
                                 activation='relu',
                                 kernel_initializer=keras.initializers.RandomNormal()))
            self.model.add(keras.layers.Dropout(rate=0.5))
        
        self.model.add(Dense(units=num_classes,
                             kernel_initializer=keras.initializers.RandomNormal()))
        self.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.Adam(lr=learning_rate),
                           metrics=['mae'])

        # Initialize Tensorboard.
        self.tensorboard_path = tensorboard_path
        self.tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=10, batch_size=32,
                                                       write_graph=False, write_grads=True, write_images=True,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None)

    def learn(self, observations, actions, rewards, iterations, batch_size=32, 
              experience_replay=True, verbose=0):
        """
        :param observations: Either a single trajectory of a list of trajectories.
        :param actions: Either a single trajectory of a list of trajectories.
        :param rewards: Either a single trajectory of a list of trajectories.
        :param iterations: The number of batches to run. If experience_replay is True, use input batch_size, if False
               batch size defaults to the entire training data.
        :param batch_size: Used in experience_replay mode.
        :param experience_replay: Let you randomly sample training data with replacement.
        :param verbose: Verbose level for TensorFlow.
        """
        ## Pre-processing
        # Convert inputs to numpy arrays to NN training.
        x, y = self.convert_input(observations, actions, rewards)
        # For metric generation.
        initial_epoch = 0

        # Train using all samples one at a time.
        if not experience_replay:
            for _ in range(iterations):
                # Epoch here actually correspond to the typical epoch.
                initial_epoch += 1
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
                    y1[0][np.isnan(y1[0])] = yp[0][np.isnan(y1[0])]
                    self.model.fit(x=x1,
                                   y=y1,
                                   verbose=verbose,
                                   epochs=initial_epoch+1,
                                   initial_epoch=initial_epoch,
                                   callbacks=[self.tensorboard],
                                   validation_data=(x, y))
        # Train using batches of random samples from all trajectories.
        else:
            # Requires manually handling epochs.
            for _ in range(iterations):
                # Epoch here doesn't mean running through the dataset one time.
                initial_epoch += 1

                idxs = np.random.randint(0, x.shape[0], batch_size)
                x_batch = x[idxs]
                y_batch_actions = y[idxs]
                
                # Predict actions not taken so we do not learn from them.
                y_batch = self.model.predict(x_batch)
                y_batch[~np.isnan(y_batch_actions)] = y_batch_actions[~np.isnan(y_batch_actions)]
                # Minmum value should be zero. Speed up by changing negatives.
                y_batch[y_batch < 0] = 0.0

                # Learn from these examples.
                self.model.fit(x=x_batch,
                               y=y_batch,
                               batch_size=batch_size,
                               verbose=verbose,
                               epochs=initial_epoch+1,
                               initial_epoch=initial_epoch,
                               callbacks=[self.tensorboard],
                               validation_data=(x_batch, y_batch))
        print('Done learning. Run:\n    tensorboard --logdir={0}\nto see your results.'.format(self.tensorboard_path))

    def action(self, observation, verbose=0):
        observations = [[observation]]
        dim_h = observation[0].shape[0]
        dim_v = observation[0].shape[1]
        this_finger = observation[1]
        invalid_action_list = []
        if this_finger[0] <= 0:
            invalid_action_list.append(0)  # Left.
        if this_finger[0] >= dim_h - 1:
            invalid_action_list.append(1)  # Right.
        if this_finger[1] >= dim_v - 1:
            invalid_action_list.append(2)  # Up.
        if this_finger[1] <= 0:
            invalid_action_list.append(3)  # Down.

        prediction = self.model.predict(self.convert_input(observations=observations)[0])
        if verbose > 0:
            print(prediction)

        for index in invalid_action_list:
            prediction[0][index] = -np.inf
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
        if type(observations) is not list:
            raise Exception('Obervations should be a list!')
        elif type(observations[0]) is list:
            list_of_lists = True
            total_frames = sum([len(inner_list) for inner_list in observations])
        elif type(observations[0]) is tuple:
            list_of_lists = False
            total_frames = len(observations[0])
        else:
            raise Exception('Elements in list observations do not have the correct type.')

        x = np.zeros((total_frames, 6, 5, 7))
        y = np.zeros((total_frames, 5))
        y[:] = np.nan

        if list_of_lists:
            counter = 0
            # Looping through episodes.
            for i in range(len(observations)):
                # Looping through steps in episodes.
                for j in range(len(observations[i])):
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
        else:
            # Looping through steps in episodes.
            for i in range(len(observations)):
                board = observations[i][0]
                finger = observations[i][1]
                x[i][:, :, 0][board == 0] = np.ones((6, 5))[board == 0]
                x[i][:, :, 1][board == 1] = np.ones((6, 5))[board == 1]
                x[i][:, :, 2][board == 2] = np.ones((6, 5))[board == 2]
                x[i][:, :, 3][board == 3] = np.ones((6, 5))[board == 3]
                x[i][:, :, 4][board == 4] = np.ones((6, 5))[board == 4]
                x[i][:, :, 5][board == 5] = np.ones((6, 5))[board == 5]
                x[i][finger[0], finger[1], 6] = 1

        if list_of_lists and actions:
            counter = 0
            # Looping through episodes.
            for i in range(len(actions)):
                # Looping through steps in episodes.
                for j in range(len(actions[i])):
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
                    counter += 1
        elif actions:
            # Looping through steps in episodes.
            for i in range(len(actions)):
                action = actions[i]
                if action == 'left':
                    y[i][0] = rewards[i]
                elif action == 'right':
                    y[i][1] = rewards[i]
                elif action == 'up':
                    y[i][2] = rewards[i]
                elif action == 'down':
                    y[i][3] = rewards[i]
                elif action == 'pass':
                    y[i][4] = rewards[i]

        return x, y


def agent01():
    return Agent01()
