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
# Basic parameters and some non-tunable hyperparameters.
p = tf.contrib.training.HParams(dim_h=6,              # The width of the board. Agent 02 only deals with 6 by 5 boards.
                                dim_v=5,              # The height of the board. Agent 02 only deals with 6 by 5 boards.
                                channels=7,           # 6 colors plus finger position.
                                action_space_size=5,  # The number of all of the possible actions (left, right, top, down, pass).
                                tensorboard_path=tensorboard_path)  # The path to where tensorboard stores its files.
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


def model_fn(x, y, mode, p, hp):
    """
    :param x: Features. x should be an numpy array with dimensions [batch_size, p.dim_h, p.dim_v, p.channels].
    :param y: Labels.
    :param mode: Train, eval or predict.
    :param p: Basic parameters and some non-tunable hyperparameters.
    :param hp: Tunable hyperparameters.
    :return:
    """
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
        x = tf.reshape(x, [-1, p.dim_h, p.dim_v, p.channels], name='x')

        # Convolution layer 1.
        conv1 = tf.layers.conv2d(
            inputs=x,  # [batch_size, 6, 5, 7]
            filters=hp.filters_conv1,
            kernel_size=[hp.kernel_conv1, hp.kernel_conv1],
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(),
            name='conv1'
        )

        # Convolution layer 2.
        conv2 = tf.layers.conv2d(
            inputs=conv1,  # [batch_size, 4, 3, 32]
            filters=hp.filters_conv2,
            kernel_size=[hp.kernel_conv2, hp.kernel_conv2],
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
            units=hp.units_dense1,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(),
            name='dense1'
        )
        dense1 = tf.layers.dropout(dense1, rate=hp.dropout, training=is_training, name='dense1_do')

        # Dense layer 2.
        dense2 = tf.layers.dense(
            inputs=dense1,  # [batch_size, 64]
            units=hp.units_dense2,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(),
            name='dense2'
        )
        dense2 = tf.layers.dropout(dense2, rate=hp.dropout, training=is_training, name='dense2_do')

        # The predictive, last dense layer.
        dense3 = tf.layers.dense(
            inputs=dense2,  # [batch_size, 64]
            units=p.action_space_size,
            kernel_initializer=tf.random_normal_initializer(),
            name='dense3'
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            return dense3
        elif mode == tf.estimator.ModeKeys.EVAL:
            pass
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'damage': dense3
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)


# The Agent 02 estimator.
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   params=hp,
                                   config=config)


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
        y[:] = np.nan
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
