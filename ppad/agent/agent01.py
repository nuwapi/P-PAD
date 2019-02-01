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
import datetime


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
        self.last_action = None

        if self.dense_layers[-1] != self.tg_shape[0]:
            raise Exception('ERROR: The last dense layer needs to have the shape of the target!')

        # Initialize models A and B.
        self.state_A, self.target_A, self.q_values_A, self.loss_A = self.initialize_model(scope='model_A')
        self.state_B, self.target_B, self.q_values_B, self.loss_B = self.initialize_model(scope='model_B')

        # Set up optimizer for model A.
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*self.optimizer.compute_gradients(self.loss_A))
        self.gradients = zip(gradients, variables)
        self.backprop_operation = self.optimizer.apply_gradients(self.gradients)

        # Set up a saver to save the graph (both models A and B).
        self.saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables.
        self.sess.run(tf.global_variables_initializer())

        # Initialize Tensorboard.
        if tensorboard_path is not None:
            # TODO: Add tensorboard.
            self.tensorboard = None
        else:
            self.tensorboard = None

    def initialize_model(self, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Define input data. The batch size is undetermined.
            state = tf.placeholder(tf.float32, shape=[None] + list(self.st_shape), name='state')
            target = tf.placeholder(tf.float32, shape=[None] + list(self.tg_shape), name='target')

            # Format: [batch, height, width, channels].
            conv_layer_in = state
            for i in range(len(self.conv_layers)):
                conv_layer_out = tf.layers.conv2d(
                    inputs = conv_layer_in,
                    filters = self.conv_layers[i][1],
                    kernel_size = (self.conv_layers[i][0], self.conv_layers[i][0]),
                    strides=(1, 1),
                    padding = 'valid',
                    kernel_initializer = tf.contrib.layers.xavier_initializer(),
                    name='model_core/conv' + str(i)
                )
                conv_layer_in = conv_layer_out
            
            relu_conv_layer = tf.nn.leaky_relu(conv_layer_out,)
            dense_layer_in = tf.contrib.layers.flatten(relu_conv_layer, scope='model_core/conv_flatten')
            for i in range(len(self.dense_layers)):
                dense_layer_out = tf.layers.dense(
                    inputs=dense_layer_in,
                    units=self.dense_layers[i],
                    activation = tf.nn.leaky_relu,
                    kernel_initializer = tf.contrib.layers.xavier_initializer(),
                    name='model_core/dense' + str(i)
                )
                dense_layer_in = dense_layer_out

            q_values = tf.nn.relu(dense_layer_out)
            loss = tf.reduce_mean(tf.squared_difference(q_values, target))

            return state, target, q_values, loss

    def train(self, boards, fingers, targets):
        """
        Train one batch.
        :param boards: Numpy array, (batch_size, 5, 6).
        :param fingers: Numpy array, (batch_size, 2).
        :param targets: Numpy array, (batch_size, 5).
        :return:
        """
        states = self.board_finger_to_state(boards, fingers)
        inputs = {self.state_A.name: states, self.target_A.name: targets}
        fetches = [self.backprop_operation, self.q_values_A, self.loss_A]
        _, q_values, loss = self.sess.run(fetches=fetches, feed_dict=inputs)

        return loss, q_values

    def predict(self, boards, fingers, model='A'):
        states = self.board_finger_to_state(boards, fingers)
        return self.predict_from_state(states, model)

    def predict_from_state(self, batch_state, model='A'):
        if model == 'A':
            state_tensor = self.state_A
            q_values_tensor = self.q_values_A
        elif model == 'B':
            state_tensor = self.state_B
            q_values_tensor = self.q_values_B
        else:
            raise Exception('ERROR: Model name has to be either A or B!')

        inputs = {state_tensor.name: batch_state}
        q_value_predictions = self.sess.run(fetches=[q_values_tensor], feed_dict=inputs)[0]

        return q_value_predictions

    def act(self, board, finger, model='A', method='max', beta=None):
        """
        Ask the agent to product an action based on the given state.
        :param board: Numpy array of size (5, 6). 1 for batch size 1.
        :param finger: Numpy array of size (2).
        :param model: A or B.
        :param method: How to turn q value array into an action.
        :param beta: The beta factor for a Boltzmann action.
        :return: The best action to take given the state.
        """
        board = np.reshape(board, tuple([1] + list(board.shape)))
        finger = np.reshape(finger, tuple([1] + list(finger.shape)))
        state = self.board_finger_to_state(board, finger)
        q_value_predictions = self.predict_from_state(state, model)[0]

        # Penalize illegal moves and last move
        if finger[0, 0] == 0:
            q_value_predictions[0] = -np.inf  # Can't go up.
        elif finger[0, 0] == self.st_shape[0] - 1:
            q_value_predictions[1] = -np.inf # Can't go down.
        if finger[0, 1] == 0:
            q_value_predictions[2] = -np.inf  # Can't go left.
        elif finger[0, 1] == self.st_shape[1] - 1:
            q_value_predictions[3] = -np.inf # Can't go right.
        
        # Penalize last move
        if self.last_action == 0: 
            q_value_predictions[1] = -np.inf # Can't go down.
        elif self.last_action == 1:
            q_value_predictions[0] = -np.inf # Can't go up.
        elif self.last_action == 2:
            q_value_predictions[3] = -np.inf # Can't go right.
        elif self.last_action == 3:
            q_value_predictions[2] = -np.inf  # Can't go left.
        
        # Choose an action according to the chosen method
        if str(method).lower() == 'max':
            action = np.argmax(q_value_predictions)
        elif str(method).lower() == 'boltzmann':
            if beta is None:
                raise Exception('ERROR: Provide a value for beta!')
            else:
                b_values = np.exp(beta*q_value_predictions)
                b_probs = b_values/sum(b_values)
                action = np.random.choice(5,p=b_probs)
        else:
            raise Exception('ERROR: Unknown action method!')
        # TODO: Epsilon greedy.

        self.last_action = action
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

    def save(self, checkpoint=None):
        if checkpoint is None:
            checkpoint = 'model.ckpt-' + str(datetime.datetime.now()).replace('-', '.').replace(':', '.').replace(' ', '.')
        self.saver.save(self.sess, checkpoint)

    def load(self, checkpoint):
        saver = tf.train.import_meta_graph(checkpoint + '.meta')
        saver.restore(self.sess, checkpoint)

    def copy_A_to_B(self, verbose=False):
        name_filters = ['filter:0', 'kernel:0', 'bias:0']
        all_nodes = [node.values() for node in tf.get_default_graph().get_operations()]
        all_names = [tensor.name for tensors in all_nodes for tensor in tensors]
        all_weight_names = []
        for name in all_names:
            for name_filter in name_filters:
                if name.find(name_filter) > -1:
                    all_weight_names.append(name)
                    break

        A_names = [name for name in all_weight_names if name.find('model_A') > -1]
        B_names = [name for name in all_weight_names if name.find('model_B') > -1]
        A_names = [name for name in A_names if name.find('model_core') > -1]
        B_names = [name for name in B_names if name.find('model_core') > -1]

        if verbose:
            for A_name, B_name in zip(A_names, B_names):
                print('Overwriting {0} with {1}'.format(B_name, A_name))

        A_tensors = [tf.get_default_graph().get_tensor_by_name(name) for name in A_names]
        B_tensors = [tf.get_default_graph().get_tensor_by_name(name) for name in B_names]
        assign_ops = [tf.assign(B_tensor, A_tensor) for A_tensor, B_tensor in zip(A_tensors, B_tensors)]
        self.sess.run(fetches=assign_ops)

    def board_finger_to_state(self, boards, fingers):
        """
        Convert a batch of states into TensorFlow usable format.
        :param boards: Numpy array, e.g. (batch_size, 5, 6).
        :param fingers: Numpy array, e.g. (batch_size, 2).
        :return: The state.
        """
        batch_size = boards.shape[0]
        # TODO: Now we fix the number of channels to 7, but in the future we should add poison, bomb etc.
        states = np.zeros((batch_size, self.st_shape[0], self.st_shape[1], self.st_shape[2]))

        # Loop through the whole batch.
        for i in range(batch_size):
            board = boards[i, :]
            finger = fingers[i, :]
            for j in range(6):
                states[i,:,:,j] = np.array(board == j)
            states[i, finger[0], finger[1], 6] = 1

        return states
