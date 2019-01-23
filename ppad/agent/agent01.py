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
import tensorflow as tf


class Agent01:
    def __init__(self,
                 sess,
                 st_shape=(5, 6, 7),
                 tg_shape=(5,),
                 conv_layers=((3, 32), (3, 32)),
                 dense_layers=(64, 5),
                 learning_rate=0.01,
                 tensorboard_path=None):
        """

        :param sess: TensorFlow session.
        :param st_shape: State shape, 5 by 6 board with 7 channels (5 colors plus heal and finger position).
        :param tg_shape: Target shape, left, right, up, down and pass.
        :param conv_layers: Tuple of kernel lengths. The length of the tuple equals to the number of conv layers to use.
        :param dense_layers: Tuple of dense layer units. The length of the tuple equals to the number of dense layers to use.
        :param learning_rate: The learning rate.
        :param tensorboard_path: Optional. Where to save tensorboard logs.
        """
        self.sess = sess
        self.st_shape = st_shape
        self.tg_shape = tg_shape
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.learning_rate = learning_rate
        self.num_conv_layers = len(conv_layers)
        self.num_dense_layers = len(dense_layers)
        self.tensorboard_path = tensorboard_path

        if self.dense_layers[-1] != self.tg_shape[0]:
            raise Exception('ERROR: The last dense layer needs to have the shape of the target!')

        # Initialize models A and B.
        self.state_A, self.target_A, self.q_value_A, self.loss_A = self.initialize_model(scope='model_A')
        self.state_B, self.target_B, self.q_value_B, self.loss_B = self.initialize_model(scope='model_B')

        # Set up optimizer for model A.
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*self.optimizer.compute_gradients(self.loss_A))
        self.gradients = zip(gradients, variables)
        self.training_operation = self.optimizer.apply_gradients(self.gradients)

        # Set up a saver to save the graph (both models A and B).
        self.saver = tf.train.Saver(tf.global_variables())

        # Initialize Tensorboard.
        if tensorboard_path is not None:
            # TODO: Add tensorboard.
            self.tensorboard = None
        else:
            self.tensorboard = None

    def initialize_model(self, scope='model_A'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Define input data. The batch size is undetermined.
            state = tf.placeholder(tf.float32, shape=[None] + list(self.st_shape), name='state')
            target = tf.placeholder(tf.float32, shape=[None] + list(self.tg_shape), name='target')

            # Format: [batch, height, width, channels].
            conv_layer_in = state
            for i in range(len(self.conv_layers)):
                in_channels = tf.shape(conv_layer_in)[-1]
                conv_layer_out = tf.nn.conv2d(
                    input=conv_layer_in,
                    filter=[self.conv_layers[i][0], self.conv_layers[i][0], in_channels, self.conv_layers[i][1]],
                    strides = [1] + [1] * len(self.st_shape),
                    padding='VALID',
                    name='model/conv' + str(i)
                )
                conv_layer_in = conv_layer_out

            dense_layer_in = tf.contrib.layers.flatten(conv_layer_out, scope='model/conv_flatten')
            for i in range(len(self.dense_layers)):
                dense_layer_out = tf.layers.dense(
                    inputs=dense_layer_in,
                    units=self.dense_layers[i],
                    name='model/dense' + str(i)
                )
                dense_layer_in = dense_layer_out

            q_value = dense_layer_out
            loss = tf.reduce_mean(tf.squared_difference(q_value, target))

            return state, target, q_value, loss

    def train(self, boards, fingers, targets):
        """
        """
        states = self.board_finger_to_state(boards, fingers)
        inputs = {self.state_A.name: states, self.target_A.name: targets}
        fetches = [self.training_operation, self.q_value_A, self.loss_A]
        _, q_value, loss = self.sess.run(fetches=fetches, feed_dict=inputs)

        return loss

    def predict(self, batch_state, batch_target, model='A'):
        if model == 'A':
            state_tensor = self.state_A
            target_tensor = self.target_A
            q_value_tensor = self.q_value_A
        elif model == 'B':
            state_tensor = self.state_B
            target_tensor = self.target_B
            q_value_tensor = self.q_value_B
        else:
            raise Exception('ERROR: Model name has to be either A or B!')

        inputs = {state_tensor.name: batch_state, target_tensor.name: batch_target}
        q_value_predictions = self.sess.run(fetches=[q_value_tensor], feed_dict=inputs)[0]

        return q_value_predictions

    def act(self, board, finger, target, model='A', method='max'):
        """
        Ask the agent to product an action based on the given state.
        :param board: Numpy array of size (1, 5, 6). 1 for batch size 1.
        :param finger: Numpy array of size (1, 2).
        :param target: Numpy array of size (1, 5).
        :param model: A or B.
        :param method: How to turn q value array into an action.
        :return: The best action to take given the state.
        """
        state = self.board_finger_to_state(board, finger)
        q_value_predictions = self.predict(state, target, model)[0]

        if str(method).lower() == 'max':
            minimum = np.min(q_value_predictions)
            if finger[0, 0] == 0:
                q_value_predictions[0] = minimum - 1  # Can't go up.
            elif finger[0, 0] == self.st_shape[0] - 1:
                q_value_predictions[1] = minimum - 1  # Can't go down.
            if finger[0, 1] == 0:
                q_value_predictions[2] = minimum - 1  # Can't go left.
            elif finger[0, 1] == self.st_shape[1] - 1:
                q_value_predictions[3] = minimum - 1  # Can't go right.

            action = np.argmax(q_value_predictions)
        else:
            raise Exception('ERROR: Unknown action method!')
        # TODO: Epsilon greedy, Boltzmann.
        # probabilities = tf.nn.softmax(q_value, axis=-1, name='softmax')

        if action == 0:
            return 'up'
        elif action == 1:
            return 'down'
        elif action == 2:
            return 'left'
        elif action == 3:
            return 'right'
        elif action == 4:
            return 'pass'
        else:
            raise Exception('Action type {0} is invalid.'.format(action))

    def copy_B_to_A(self):
        pass

    def board_finger_to_state(self, boards, fingers):
        """
        Convert a batch of states into TensorFlow usable format.
        :param boards: Numpy array, e.g. (batch_size, 5, 6).
        :param fingers: Numpy array, e.g. (batch_size, 2)
        :return: The state.
        """
        batch_size = len(boards)
        # TODO: Now we fix the number of channels to 7, but in the future we should add poison, bomb etc.
        states = np.zeros((batch_size, self.st_shape[0], self.st_shape[1], self.st_shape[2]))

        # Loop through the whole batch.
        for i in range(batch_size):
            board = boards[i, :]
            finger = fingers[i, :]
            states[i, :, :, 0][board == 0] = np.ones((5, 6))[board == 0]
            states[i, :, :, 0][board == 1] = np.ones((5, 6))[board == 1]
            states[i, :, :, 0][board == 2] = np.ones((5, 6))[board == 2]
            states[i, :, :, 0][board == 3] = np.ones((5, 6))[board == 3]
            states[i, :, :, 0][board == 4] = np.ones((5, 6))[board == 4]
            states[i, :, :, 0][board == 5] = np.ones((5, 6))[board == 5]
            states[i, finger[0], finger[1], 6] = 1

        return states
