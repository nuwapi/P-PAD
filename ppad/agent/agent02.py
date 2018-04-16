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

import datetime
import numpy as np
import os
import pathlib
import tensorflow as tf

# Configure logging.
tf.logging.set_verbosity(tf.logging.INFO)

# Set up parameters.
tensorboard_path = os.path.join(str(pathlib.Path.home()),
                                'ppad-tensorboard_'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

# Tunable hyperparameters.
hp = tf.contrib.training.HParams(filters_conv1=32,
                                 kernel_conv1=3,
                                 filters_conv2=32,
                                 kernel_conv2=2,
                                 units_dense1=64,
                                 units_dense2=64,
                                 dropout=0.5,
                                 learning_rate=0.01)

# Tensorflow configurations.
config = tf.estimator.RunConfig()
# We don't really set any config up yet. Could be useful later if we want to monitor the training or save models.


def model_fn(features, labels, mode, params):
    """
    :param features: Features. x should be an numpy array with dimensions [batch_size, p.dim_h, p.dim_v, p.channels].
    :param y: Labels.
    :param mode: Train, eval or predict.
    :param p: Basic parameters and some non-tunable hyperparameters.
    :param params: Tunable hyperparameters.
    :return:
    """
    # Basic parameters and some non-tunable hyperparameters.
    p = tf.contrib.training.HParams(dim_h=6,  # The width of the board. Agent 02 only deals with 6 by 5 boards.
                                    dim_v=5,  # The height of the board. Agent 02 only deals with 6 by 5 boards.
                                    channels=7,  # 6 colors plus finger position.
                                    action_space_size=5,
                                    # The number of all of the possible actions (left, right, top, down, pass).
                                    tensorboard_path=tensorboard_path)  # The path to where tensorboard stores its files.
    # Set up constant tensors.
    one = tf.constant(1, dtype=tf.float32)
    zero = tf.constant(0, dtype=tf.float32)
    dim_h = tf.constant(p.dim_h, dtype=tf.float32)
    dim_v = tf.constant(p.dim_v, dtype=tf.float32)

    # Always reuse variables if scope 'agent02' is not empty.
    # If we did not initialize all of the necessary variables Tensorflow will throw an error.
    reuse = True
    if len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent02')) == 0:
        reuse = False
    # Use the mode type to determine whether to use dropout.
    is_training = False
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True

    # Define a scope for reusing the neural network weights.
    with tf.variable_scope('agent02', reuse=reuse):
        # Input.
        features = tf.reshape(features, [-1, p.dim_h, p.dim_v, p.channels], name='x')

        # Convolution layer 1.
        conv1 = tf.layers.conv2d(
            inputs=features,  # [batch_size, 6, 5, 7]
            filters=params.filters_conv1,
            kernel_size=[params.kernel_conv1, params.kernel_conv1],
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(),
            name='conv1'
        )

        # Convolution layer 2.
        conv2 = tf.layers.conv2d(
            inputs=conv1,  # [batch_size, 4, 3, 32]
            filters=params.filters_conv2,
            kernel_size=[params.kernel_conv2, params.kernel_conv2],
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(),
            name='conv2'
        )

        # Flatten the layers.
        flat = tf.contrib.layers.flatten(
            inputs=conv2,  # [batch_size, 3, 2, 32]
            name='flatten'
        )

        # Dense layer 1.
        dense1 = tf.layers.dense(
            inputs=flat,  # [batch_size, 3*2*32=192]
            units=params.units_dense1,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(),
            name='dense1'
        )
        dense1 = tf.layers.dropout(dense1, rate=params.dropout, training=is_training, name='dense1_do')

        # Dense layer 2.
        dense2 = tf.layers.dense(
            inputs=dense1,  # [batch_size, 64]
            units=params.units_dense2,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(),
            name='dense2'
        )
        dense2 = tf.layers.dropout(dense2, rate=params.dropout, training=is_training, name='dense2_do')

        # The predictive, last dense layer.
        dense3 = tf.layers.dense(
            inputs=dense2,  # [batch_size, 64]
            units=p.action_space_size,
            kernel_initializer=tf.random_normal_initializer(),
            name='dense3'
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            pass
        elif mode == tf.estimator.ModeKeys.EVAL:
            raise Exception('Evaluation mode is not yet available.')
        elif mode == tf.estimator.ModeKeys.PREDICT:
            # Get the finger position.
            this_finger = tf.where(tf.equal(features[0], one))[0]
            # See which actions cannot be taken.
            invalid_action_list = []
            if tf.less_equal(this_finger[0], zero):
                invalid_action_list.append(0)  # Left.
            if tf.greater_equal(this_finger[0], dim_h - 1):
                invalid_action_list.append(1)  # Right.
            if tf.greater_equal(this_finger[1], dim_v - 1):
                invalid_action_list.append(2)  # Up.
            if tf.less_equal(this_finger[1], zero):
                invalid_action_list.append(3)  # Down.

            # for index in invalid_action_list:
            #     prediction[0][index] = -np.inf
            # action_type = np.unravel_index(np.argmax(prediction[0], axis=None), prediction[0].shape)
            # action_type = int(action_type[0])
            # if action_type == 0:
            #     action = 'left'
            # elif action_type == 1:
            #     action = 'right'
            # elif action_type == 2:
            #     action = 'up'
            # elif action_type == 3:
            #     action = 'down'
            # elif action_type == 4:
            #     action = 'pass'
            # else:
            #     raise Exception('Action type {0} is invalid.'.format(action_type))

            predictions = {
                'all_damages': dense3,
                'action': "NULL"
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)


# The Agent 02 estimator.
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   params=hp,
                                   config=config)


def input_fn(observations=None,
             actions=None,
             rewards=None,
             mode=tf.estimator.ModeKeys.PREDICT,
             sample_with_replacement=False):
    """
    Convert the output of the environment into Tensorflow usable input.
    Here we concatenate all of the frames of all of the trajectories.
    During training, there is no concept of trajectories, just a bunch of frames.
    :param observations: observations[i] or observations[i][j] = (np_array(6,5), np_array(2)).
    :param actions: actions[i] or actions[i][j] = string.
    :param rewards: rewards[i] or rewards[i][j] = float.
    :param mode: TRAIN, EVAL or PREDICT.
    :param sample_with_replacement: Trues means randomly provide samples, false means sequentially provide samples.
    :return:
    """
    # 1. Set up some internal parameters.
    # Batch size has to be one, because we do a prediction every time a sample is given.
    batch_size = 1

    # 2. Converting the data.
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

    # If we want to imitate experience replay.
    if sample_with_replacement:
        # TODO: Write this!
        pass

    # 3. Turning data into a Dataset and initialize the dataset.
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Shuffle data in the training mode.
        dataset = dataset.shuffle(buffer_size=len(x))
        # Repeat the data indefinitely. Every repeat has a different shuffling.
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    x_sample, y_sample = iterator.get_next()

    return x_sample, y_sample
