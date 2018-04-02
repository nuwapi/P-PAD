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


class Agent02:
    """
    Agent02 focuses on 6 by 5 boards. Agent 02 is written in Tensorflow.
    """
    def __init__(self, learning_rate=0.01, tensorboard_path='~/ppad_logs/1'):
        tf.logging.set_verbosity(tf.logging.INFO)

        # Neural net specifications.
        self.dim_h = 6
        self.dim_v = 5
        self.channels = 7
        self.filters1 = 32
        self.kernel_len1 = 3
        self.filters2 = 32
        self.kernel_len2 = 2
        self.dense_units1 = 64
        self.dense_units2 = 64
        self.action_space_size = 5
        self.dropout = 0.5
        self.learning_rate = learning_rate
        self.tensorboard_path = tensorboard_path

        # Set up TF's high-level estimator API.
        self.initialized = False
        self.model = tf.estimator.Estimator(self.model_fn)

    def CNN(self, x_in, reuse, is_training):
        """
        :param x_in: In put data in dimensions  [batch_size, self.dim_h, self.dim_v, self.channels].
        :param reuse: True for first time initialization.
        :param is_training:
        :return:
        """
        # Define a scope for reusing the neural network weights.
        with tf.variable_scope('CNN', reuse=reuse):
            # Input.
            x = tf.reshape(x_in, [-1, self.dim_h, self.dim_v, self.channels], name='x')

            # Convolution layer 1.
            conv1 = tf.layers.conv2d(
                inputs=x,      # [batch_size, 6, 5, 7]
                filters=self.filters1,
                kernel_size=[self.kernel_len1, self.kernel_len1],
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(),
                name='conv1'
            )

            # Convolution layer 2.
            conv2 = tf.layers.conv2d(
                inputs=conv1,  # [batch_size, 4, 3, 32]
                filters=self.filters2,
                kernel_size=[self.kernel_len2, self.kernel_len2],
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
                units=self.dense_units1,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(),
                name='dense1'
            )
            dense1 = tf.layers.dropout(dense1, rate=self.dropout, training=is_training, name='dense1_do')

            # Dense layer 2.
            dense2 = tf.layers.dense(
                inputs=dense1,  # [batch_size, 64]
                units=self.dense_units2,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(),
                name='dense2'
            )
            dense2 = tf.layers.dropout(dense2, rate=self.dropout, training=is_training, name='dense2_do')

            # The predictive last dense layer.
            dense3 = tf.layers.dense(
                inputs=dense2,  # [batch_size, 64]
                units=self.action_space_size,
                kernel_initializer=tf.random_normal_initializer(),
                name='predict'
            )

        return dense3

    def model_fn(self, x, y, mode):
        if self.initialized is False:
            CNN_train = self.CNN(x, reuse=False, is_training=True)
            self.initialized = True
        else:
            CNN_train = self.CNN(x, reuse=True, is_training=True)
        CNN_inference = self.CNN(x, reuse=True, is_training=False)

        if mode is tf.estimator.ModeKeys.TRAIN:
            pass
        elif mode is tf.estimator.ModeKeys.EVAL:
            pass
        elif mode is tf.estimator.ModeKeys.PREDICT:
            pred_classes = CNN_inference

        else:
            raise Exception('Invalid tf.estimator.ModeKeys!')



        pred_classes = tf.argmax(CNN_predict, axis=1)
        pred_probas = tf.nn.softmax(CNN_predict)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs



        #
        #
        # raw_y = tf.placeholder(tf.float32, [None, self.action_space_size], name='raw_y')
        # y = tf.reshape(raw_y, [-1, self.action_space_size], name='y')
        # # Loss function and optimizer.
        # loss = tf.losses.mean_squared_error(
        #     labels=y,
        #     predictions=self.dense3,
        # )
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #
        #
        #
        # # Configure the Training Op (for TRAIN mode)
        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #     train_op = optimizer.minimize(
        #         loss=loss,
        #         global_step=tf.train.get_global_step())
        #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        #
        #
        #
        # self.model.compile(loss=keras.losses.mean_squared_error,
        #                    optimizer=keras.optimizers.Adam(lr=learning_rate),
        #                    metrics=['accuracy'])
        # # Initialize Tensorboard.
        # self.tensorboard_path = tensorboard_path
        # self.tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=0, batch_size=32,
        #                                                write_graph=True, write_grads=False, write_images=False,
        #                                                embeddings_freq=0, embeddings_layer_names=None,
        #                                                embeddings_metadata=None)
##################################

        # predictions = {
        #     # Generate predictions (for PREDICT and EVAL mode)
        #     "classes": tf.argmax(input=logits, axis=1),
        #     # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        #     # `logging_hook`.
        #     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        # }
        #
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        #
        # # Calculate Loss (for both TRAIN and EVAL modes)
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        #
        # # Configure the Training Op (for TRAIN mode)
        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        #     train_op = optimizer.minimize(
        #         loss=loss,
        #         global_step=tf.train.get_global_step())
        #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        #
        # # Add evaluation metrics (for EVAL mode)
        # eval_metric_ops = {
        #     "accuracy": tf.metrics.accuracy(
        #         labels=labels, predictions=predictions["classes"])}
        # return tf.estimator.EstimatorSpec(
        #     mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

###################################################


    def learn(self, observations, actions, rewards, iterations, batch_size=32, 
              experience_replay=True):
        ## Pre-processing
        # Convert inputs to numpy arrays to NN training.
        x, y = self.convert_input(observations, actions, rewards)
        
        # Train using all samples one at a time.
        if not experience_replay:
            for _ in range(iterations):
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
                                   verbose=0,
                                   callbacks=[self.tensorboard])
        # Train using batches of random samples from all trajectories.
        else:
            # Initialize arrays for batch data.
            x_batch = np.zeros((batch_size, 6, 5, 7))
            y_batch = np.zeros((batch_size, 5))
            
            # Requires manually handling epochs.
            for _ in range(iterations):
                idxs = np.random.randint(0,x.shape[0],batch_size)
                x_batch = x[idxs]
                y_batch_actions = y[idxs]
                
                # Predict actions not taken so we do not learn from them.
                y_batch = self.model.predict(x_batch)
                y_batch[~np.isnan(y_batch_actions)] = y_batch_actions[~np.isnan(y_batch_actions)]
                
                # Learn from these examples.
                self.model.fit(x=x_batch,
                               y=y_batch,
                               batch_size=batch_size,
                               verbose=0,
                               callbacks=[self.tensorboard])
        print('Done learning. Run:\n    tensorboard --logdir={0}\nto see your results.'.format(self.tensorboard_path))

    def action(self, observation):
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


def agent02():
    return Agent02()
