import tensorflow as tf
import os

import config
from data_provider import Datasets
import utils

tf.disable_eager_execution()


class Model(object):
    def __init__(self, session, trainable=True, prediction=False):
        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.batch_size = config.BATCH_SIZE
        self.sess = session
        self.model_dir = config.MODEL_DIR
        self.trainable = trainable
        self.prediction = prediction
        self.num_epochs = config.EPOCHS

        # Building model
        self._define_input()
        self._build_model()

        if not prediction:
            self._define_loss()
            # Learning rate and train op
            # learning_rate = tf.train.exponential_decay(learning_rate=config.INIT_LR, global_step=self.global_step,
            # decay_steps=config.DECAY_STEP, decay_rate=config.DECAY_LR, staircase=True)
            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.total_loss,
                                                                                  global_step=self.global_step)
            # Input data
            self.data = Datasets(trainable=self.trainable, test_data_type='public_test')

        # Init checkpoints
        self.saver_all = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        self.checkpoint_path = os.path.join(self.model_dir, 'model.ckpt')
        ckpt = tf.train.get_checkpoint_state(self.model_dir)

        if ckpt:
            print('Reading model parameters from %s', ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('Created model with fresh parameters.')
            self.sess.run(tf.global_variables_initializer())

    def _define_input(self):
        self.input_images = tf.placeholder(tf.float32, [None, config.IMAGE_SIZE, config.IMAGE_SIZE, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase_train = tf.placeholder(tf.bool)
        if not self.prediction:
            self.input_labels = tf.placeholder(tf.float32, [None, 7])
            self.input_indexes = tf.placeholder(tf.float32, [None])

    def _build_model(self):
        # Extract features
        x = utils.VGG_ConvBlock('Block1', self.input_images, 1, 32, 2, 1, self.phase_train)
        x = utils.VGG_ConvBlock('Block2', x, 32, 64, 2, 1, self.phase_train)
        x = utils.VGG_ConvBlock('Block3', x, 64, 128, 2, 1, self.phase_train)
        x = utils.VGG_ConvBlock('Block4', x, 128, 256, 3, 1, self.phase_train)

        # Smile branch
        smile_fc1 = utils.FC('smile_fc1', x, 256, self.keep_prob)
        smile_fc2 = utils.FC('smile_fc2', smile_fc1, 256, self.keep_prob)
        self.y_smile_conv = utils.FC('smile_softmax', smile_fc2, 2, self.keep_prob, 'softmax')

        # Gender branch
        gender_fc1 = utils.FC('gender_fc1', x, 256, self.keep_prob)
        gender_fc2 = utils.FC('gender_fc2', gender_fc1, 256, self.keep_prob)
        self.y_gender_conv = utils.FC('gender_softmax', gender_fc2, 2, self.keep_prob, 'softmax')

        # Age branch
        age_fc1 = utils.FC('age_fc1', x, 256, self.keep_prob)
        age_fc2 = utils.FC('age_fc2', age_fc1, 256, self.keep_prob)
        self.y_age_conv = utils.FC('age_softmax', age_fc2, 5, self.keep_prob, 'softmax')

    def _define_loss(self):
        self.smile_mask = tf.cast(tf.equal(self.input_indexes, 1.0), tf.float32)
        self.age_mask = tf.cast(tf.equal(self.input_indexes, 3.0), tf.float32)
        self.gender_mask = tf.cast(tf.equal(self.input_indexes, 4.0), tf.float32)

        self.y_smile = self.input_labels[:, :2]
        self.y_age = self.input_labels[:, :5]
        self.y_gender = self.input_labels[:, :2]

        # Extra variables
        smile_correct_prediction = tf.equal(tf.argmax(self.y_smile_conv, 1), tf.argmax(self.y_smile, 1))
        age_correct_prediction = tf.equal(tf.argmax(self.y_age_conv, 1), tf.argmax(self.y_age, 1))
        gender_correct_prediction = tf.equal(tf.argmax(self.y_gender_conv, 1), tf.argmax(self.y_gender, 1))

        self.smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32) * self.smile_mask)
        self.age_true_pred = tf.reduce_sum(tf.cast(age_correct_prediction, dtype=tf.float32) * self.age_mask)
        self.gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32) * self.gender_mask)

        self.smile_cross_entropy = tf.reduce_mean(
            tf.reduce_sum(-self.y_smile * tf.math.log(tf.clip_by_value(tf.nn.softmax(self.y_smile_conv), 1e-10, 1.0)),
                          axis=1) * self.smile_mask)

        self.age_cross_entropy = tf.reduce_mean(
            tf.reduce_sum(-self.y_age * tf.math.log(tf.clip_by_value(tf.nn.softmax(self.y_age_conv), 1e-10, 1.0)),
                          axis=1) * self.age_mask)

        self.gender_cross_entropy = tf.reduce_mean(
            tf.reduce_sum(-self.y_gender * tf.math.log(tf.clip_by_value(tf.nn.softmax(self.y_gender_conv), 1e-10, 1.0)),
                          axis=1) * self.gender_mask)

        # Add l2 regularizer
        l2_loss = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                l2_loss.append(tf.nn.l2_loss(var))

        self.l2_loss = config.WEIGHT_DECAY * tf.add_n(l2_loss)

        self.total_loss = self.smile_cross_entropy + self.gender_cross_entropy + self.l2_loss + self.age_cross_entropy

    @staticmethod
    def count_trainable_params():
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    def train(self):
        for epoch in range(self.num_epochs):
            smile_nb_true_pred = 0
            age_nb_true_pred = 0
            gender_nb_true_pred = 0

            print("=======================================================================")
            print('Epoch %d/%d: ' % (epoch + 1, config.EPOCHS))
            for batch_image, batch_label, batch_index in self.data.gen():
                feed_dict = {self.input_images: batch_image,
                             self.input_labels: batch_label,
                             self.input_indexes: batch_index,
                             self.keep_prob: 0.5,
                             self.phase_train: self.trainable
                             }

                ttl, sml, agl, gel, l2l, _ = self.sess.run([self.total_loss, self.smile_cross_entropy,
                                                            self.age_cross_entropy,
                                                            self.gender_cross_entropy, self.l2_loss,
                                                            self.train_step], feed_dict=feed_dict)

                smile_nb_true_pred += self.sess.run(self.smile_true_pred, feed_dict=feed_dict)
                age_nb_true_pred += self.sess.run(self.age_true_pred, feed_dict=feed_dict)
                gender_nb_true_pred += self.sess.run(self.gender_true_pred, feed_dict=feed_dict)

                print('smile_loss: %.2f, age_loss: %.2f, gender_loss: %.2f, l2_loss: %.2f, total_loss: %.2f\r' % (
                    sml, agl, gel, l2l, ttl), end="")

            smile_nb_train = len(self.data.smile_train) * 10
            age_nb_train = len(self.data.age_train)
            gender_nb_train = len(self.data.gender_train)

            smile_train_acc = smile_nb_true_pred * 1.0 / smile_nb_train
            age_train_acc = age_nb_true_pred * 1.0 / age_nb_train
            gender_train_acc = gender_nb_true_pred * 1.0 / gender_nb_train

            print('\n')
            print('Smile task train accuracy: ', str(smile_train_acc * 100))
            print('Age task train accuracy: ', str(age_train_acc * 100))
            print('Gender task train accuracy: ', str(gender_train_acc * 100))

            self.saver_all.save(self.sess, self.model_dir + '/model.ckpt')

    def test(self):
        # Evaluate model on the test data

        smile_nb_true_pred = 0
        age_nb_true_pred = 0
        gender_nb_true_pred = 0

        for batch_image, batch_label, batch_index in self.data.gen():
            feed_dict = {self.input_images: batch_image,
                         self.input_labels: batch_label,
                         self.input_indexes: batch_index,
                         self.keep_prob: 1,
                         self.phase_train: self.trainable}

            smile_nb_true_pred += self.sess.run(self.smile_true_pred, feed_dict)
            age_nb_true_pred += self.sess.run(self.age_true_pred, feed_dict)
            gender_nb_true_pred += self.sess.run(self.gender_true_pred, feed_dict)

        smile_nb_test = len(self.data.smile_test)
        age_nb_test = len(self.data.age_test)
        gender_nb_test = len(self.data.gender_test)

        smile_test_accuracy = smile_nb_true_pred * 1.0 / smile_nb_test
        gender_test_accuracy = gender_nb_true_pred * 1.0 / gender_nb_test
        age_test_accuracy = age_nb_true_pred * 1.0 / age_nb_test

        print('\nResult: ')
        print('Smile task test accuracy: ' + str(smile_test_accuracy * 100))
        print('Gender task test accuracy: ' + str(gender_test_accuracy * 100))
        print('Age task test accuracy: ' + str(age_test_accuracy * 100))

    def predict(self, image):
        # M is MALE and F is Female
        SMILE_DICT = {0: 'Not Smile', 1: 'Smile'}
        GENDER_DICT = {0: 'M', 1: 'F'}
        AGE_DICT = {0: '1-13', 1: '14-23', 2: '24-39', 3: '40-55', 4: '56-80'}
        labels = []

        feed_dict = {self.input_images: image,
                     self.keep_prob: 1,
                     self.phase_train: self.trainable}

        smile_prediction_idx = self.sess.run(tf.argmax(self.y_smile_conv, axis=1), feed_dict=feed_dict)
        age_prediction_idx = self.sess.run(tf.argmax(self.y_age_conv, axis=1), feed_dict=feed_dict)
        gender_prediction_idx = self.sess.run(tf.argmax(self.y_gender_conv, axis=1), feed_dict=feed_dict)

        for i in range(len(smile_prediction_idx)):
            smile_label = SMILE_DICT[smile_prediction_idx[i]]
            age_label = AGE_DICT[age_prediction_idx[i]]
            gender_label = GENDER_DICT[gender_prediction_idx[i]]

            labels.append((smile_label, age_label, gender_label))

        return labels
