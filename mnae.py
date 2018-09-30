import tensorflow as tf
import pandas as pd
import numpy as np
import os
from visualizer import plot_metrics

from constants import NP_SEED, TF_SEED
tf.set_random_seed(TF_SEED)
np.random.seed(NP_SEED)

import util

class NN:
    def __init__(self, session: tf.Session, attr_num_list, dropout_keep_prob, lr, lr_decay, name):

        self.name = name
        self.sess = session
        self.attr_num_list = attr_num_list
        self.dropout_keep_prob = dropout_keep_prob

        with tf.name_scope(name):

            self.x = tf.placeholder(tf.float32, [None, 333], name='x')
            self.nan_mask = tf.placeholder(tf.float32, [None, 333])
            self.y = tf.placeholder(tf.float32, [None, 333], name='y')

            self.is_training = tf.placeholder(tf.bool)
            global_step = tf.Variable(0, trainable=False)
            decaying_lr = tf.train.exponential_decay(lr, global_step, 200, lr_decay, staircase=True)

            self.tf_dropout_keep_prob = tf.placeholder(tf.float32)

            X = self.x
            flat_output = self.hidden_layers(X)  # (-1, 333)

            # 각각의 feature의 attributes 개수로 split한다.
            pred_split = tf.split(flat_output, attr_num_list, axis=1)
            y_split = tf.split(self.y, attr_num_list, axis=1)

            cost_list = []
            for i in range(len(attr_num_list)):
                if attr_num_list[i] == 1:   # case: 수치형 변수
                    cost_list.append(
                        tf.losses.mean_squared_error(y_split[i], pred_split[i]))
                else:                       # case: 범주형 변수
                    cost_list.append(tf.losses.softmax_cross_entropy(
                        y_split[i], pred_split[i]))

            def output_function(output_split):
                output_list = []
                for i in range(len(attr_num_list)):
                    if attr_num_list[i] == 1:   # case: 수치형 변수
                        output_list.append(output_split[i])
                    else:                       # case: 범주형 변수
                        output_list.append(tf.nn.softmax(output_split[i]))
                return tf.concat(output_list, axis=1)

            self.output_op = output_function(pred_split)
            # loss
            self.num_loss = tf.reduce_mean(cost_list[:5])
            self.cat_loss = tf.reduce_mean(cost_list[5:])
            self.loss = self.num_loss + self.cat_loss

            self.optimizer = tf.train.AdamOptimizer(decaying_lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)


    def hidden_layers(self, x):
        """
        x: shape=[-1, 333]
        :return: x_hat: shape=[-1, 333])
        """
        raise NotImplementedError("YOU MUST IMPLEMENT hidden_layers() FUNCTION")
        pass


    def run_batch(self, x, nan_mask, batch_size, is_training):
        """
        :param x: train or validation or test data

        :return: loss, num_loss, cat_loss, x_pred
        """

        # BATCH_SIZE로 나누어 떨어지지 않는 마지막 batch도 사용함
        steps_per_epoch = len(x) // batch_size
        if len(x) % batch_size != 0:
            steps_per_epoch += 1

        loss = 0
        num_loss = 0
        cat_loss = 0
        x_preds = []
        if is_training:
            for j in range(steps_per_epoch):
                batch_x = x[j*batch_size: (j+1)*batch_size]
                batch_nan_mask = nan_mask[j*batch_size: (j+1)*batch_size]
                _, _loss, _num_loss, _cat_loss, x_pred = self.sess.run(
                                [self.train_op, self.loss, self.num_loss, self.cat_loss, self.output_op],
                                feed_dict={
                                    self.x: batch_x * batch_nan_mask,
                                    self.nan_mask: batch_nan_mask,
                                    self.y: batch_x,
                                    self.tf_dropout_keep_prob: self.dropout_keep_prob,
                                    self.is_training: True})

                loss += _loss
                num_loss += _num_loss
                cat_loss += _cat_loss
                x_preds.append(x_pred)
        else:
            for j in range(steps_per_epoch):
                batch_x = x[j * batch_size: (j + 1) * batch_size]
                batch_nan_mask = nan_mask[j*batch_size: (j+1)*batch_size]
                _, _loss, _num_loss, _cat_loss, x_pred = self.sess.run(
                    [self.train_op, self.loss, self.num_loss, self.cat_loss, self.output_op],
                    feed_dict={
                        self.x: batch_x * batch_nan_mask,
                        self.nan_mask: batch_nan_mask,
                        self.y: batch_x,
                        self.tf_dropout_keep_prob: 1.0,
                        self.is_training: False})

                loss += _loss
                num_loss += _num_loss
                cat_loss += _cat_loss
                x_preds.append(x_pred)

        x_preds = np.concatenate(x_preds)
        return loss / steps_per_epoch, num_loss / steps_per_epoch, cat_loss / steps_per_epoch, x_preds

    def problem_predict(self, x, nan_mask):
        print('> Problem Prediction... ')
        test_pred = self.sess.run(self.output_op, feed_dict={
            self.x: x,
            self.nan_mask: nan_mask,
            self.tf_dropout_keep_prob: 1.0,
            self.is_training: False})

        assert not np.isnan(np.min(test_pred))

        return test_pred

    def train(self, etler, drop_num_cols, drop_cat_cols, batch_size, total_epoch, train_all_data=False):

        # data to use
        train_input_df = etler.train_input_df
        val_input_df = etler.val_input_df
        test_input_df = etler.test_input_df
        problem_input_df = etler.problem_input_df
        result_df = etler.result_df

        val_nan_mask = etler.val_nan_mask
        test_nan_mask = etler.test_nan_mask
        val_nan_pos = etler.val_nan_pos
        test_nan_pos = etler.test_nan_pos

        problem_nan_mask = problem_input_df.notnull().values.astype(float)  # nan -> 0, else -> 1
        problem_input_df = problem_input_df.fillna(0)

        train_data = train_input_df.values
        val_data = val_input_df.values
        test_data = test_input_df.values
        problem_data = problem_input_df.values

        self.metrics = {'train': {'loss': [], 'score': []},
                        'val':   {'loss': [], 'score': []},
                        'test':  {'loss': [], 'score': []}}

        self.predicts = {'train':  {'x_pred': []},
                         'val':    {'x_pred': []},
                         'test':   {'x_pred': []},
                         'problem':{'x_pred': []}}

        for i in range(total_epoch):
            print('[NAME: {}, EPOCH: {}]'.format(self.name, i))
            print('> Train...')
            train_nan_mask, train_nan_pos = etler.gen_random_nan_mask(len(train_data), drop_num_cols, drop_cat_cols)
            train_loss, train_num_loss, train_cat_loss, train_pred = self.run_batch(train_data, train_nan_mask,
                                                                        batch_size, is_training=True)
            train_acc, train_score = util.calc_metric(train_data, train_pred, train_nan_pos, etler)

            print('> Validation...')

            val_loss, val_num_loss, val_cat_loss, val_pred = self.run_batch(val_data, val_nan_mask,
                                                                        batch_size, is_training=train_all_data)
            val_acc, val_score = util.calc_metric(val_data, val_pred, val_nan_pos, etler)
            print(val_score)

            print('> Test...')
            test_loss, test_num_loss, test_cat_loss, test_pred = self.run_batch(test_data, test_nan_mask,
                                                                        batch_size, is_training=train_all_data)
            test_acc, test_score = util.calc_metric(test_data, test_pred, test_nan_pos, etler)
            print(test_score)


            print('[train] loss: {:.4} num_loss: {:.4} cat_loss: {:.4}'.format(train_loss, train_num_loss, train_cat_loss))
            print('score: {:.4} num_score: {:.4} cat_score: {:.4}'.format(train_score.mean(),
                                                                          train_score[etler.num_vars].mean(),
                                                                          train_score[etler.cat_vars].mean()))
            print('[val]   loss:{:.4} num_loss:{:.4} cat_loss:{:.4}'.format(val_loss, val_num_loss, val_cat_loss))
            print('score: {:.4} num_score: {:.4} cat_score: {:.4}'.format(val_score.mean(),
                                                                          val_score[etler.num_vars].mean(),
                                                                          val_score[etler.cat_vars].mean()))
            print('[test]  loss:{:.4} num_loss:{:.4} cat_loss:{:.4}'.format(test_loss, test_num_loss, test_cat_loss))
            print('score: {:.4} num_score: {:.4} cat_score: {:.4}'.format(test_score.mean(),
                                                                          test_score[etler.num_vars].mean(),
                                                                          test_score[etler.cat_vars].mean()))
            print()

            problem_pred = self.problem_predict(problem_data, problem_nan_mask)

            # problem 데이터 예측 결과를 format에 맞게 만들기
            problem_imputed_df = pd.DataFrame(np.array(problem_pred), columns=problem_input_df.columns)
            problem_imputed_df = etler.generate_output_df(problem_imputed_df)
            result_df = util.fill_result_df(result_df, problem_imputed_df)

            # 정답지를 result/result_#epoch.csv 에 저장
            if not os.path.exists('result'):
                os.mkdir('result')
            result_df.to_csv(os.path.join('result', 'result_{}epoch.csv'.format(i)), index=False, encoding='cp949')

            self.metrics['train']['loss'].append(train_loss)
            self.metrics['train']['score'].append(train_score.mean())
            self.metrics['val']['loss'].append(val_loss)
            self.metrics['val']['score'].append(val_score.mean())
            self.metrics['test']['loss'].append(test_loss)
            self.metrics['test']['score'].append(test_score.mean())

            self.predicts['train']['x_pred'].append(train_pred)
            self.predicts['val']['x_pred'].append(val_pred)
            self.predicts['test']['x_pred'].append(test_pred)
            self.predicts['problem']['x_pred'].append(problem_pred)

            plot_metrics(**self.metrics)


class Dense(NN):
    def __init__(self, session: tf.Session, attr_num_list, dropout_keep_prob, lr, lr_decay, name):
        super().__init__(session, attr_num_list, dropout_keep_prob, lr, lr_decay, name)

    def hidden_layers(self, x):

        fc1 = tf.contrib.layers.fully_connected(x, 1024, activation_fn=tf.nn.relu)
        fc1 = tf.nn.dropout(fc1, self.dropout_keep_prob)

        fc2 = tf.contrib.layers.fully_connected(fc1, 1024, activation_fn=tf.nn.relu)
        fc2 = tf.nn.dropout(fc2, self.dropout_keep_prob)

        fc3 = tf.contrib.layers.fully_connected(fc2, 1024, activation_fn=tf.nn.relu)
        fc3 = tf.nn.dropout(fc3, self.dropout_keep_prob)

        output = tf.contrib.layers.fully_connected(fc3, 333, activation_fn=tf.nn.relu)

        return output


class NMAE0(NN):
    def __init__(self, session: tf.Session, attr_num_list, dropout_keep_prob, lr, lr_decay, name):
        super().__init__(session, attr_num_list, dropout_keep_prob, lr, lr_decay, name)

    def hidden_layers(self, x):

        # nan mask conv
        x = tf.stack([x, self.nan_mask], axis=2)  # (-1, 333, 2)
        x = tf.expand_dims(x, -1)  # (-1, 333, 2, 1)

        nan_mask_conv = tf.layers.conv2d(x, filters=8, kernel_size=[1, 2],
                                        strides=[1, 1], activation=tf.nn.relu)  # (-1, 333, 1, f)

        # feature conv
        feature_list = []
        splited_feature = tf.split(nan_mask_conv, self.attr_num_list, axis=1)
        for feature in splited_feature:
            feature = tf.transpose(feature, [0, 1, 3, 2])
            feature_conv = tf.layers.conv2d(feature, filters=16, kernel_size=[feature.shape[1], feature.shape[2]],
                                       strides=[1, 1], activation=tf.nn.relu)  # (-1, 1, 1, f)
            feature_list.append(tf.reshape(feature_conv, (-1, feature_conv.shape[3])))

        # related feature conv
        feature_5_6 = tf.stack([feature_list[5], feature_list[6]], axis=2)  # (-1, f, 2) 주야, 요일
        feature_7_8 = tf.stack([feature_list[7], feature_list[8]], axis=2)  # (-1, f, 2) 지시도, 시군구
        feature_9_10 = tf.stack([feature_list[9], feature_list[10]], axis=2)  # (-1, f, 2) 사고유형대, 중
        feature_12_13 = tf.stack([feature_list[12], feature_list[13]], axis=2)  # (-1, f, 2) 도로형태 대, 중
        feature_14_15 = tf.stack([feature_list[14], feature_list[15]], axis=2)  # (-1, f, 2) 당사자1, 2

        related_features = tf.stack([feature_5_6, feature_7_8, feature_9_10, feature_12_13, feature_14_15], axis=3)  # (-1, f, 2, 5)
        related_features = tf.expand_dims(related_features, axis=-1)  # (-1, f, 2, 5, 1)

        related_feature_conv = tf.layers.conv3d(related_features, filters=16,
                         kernel_size=[related_features.shape[1], related_features.shape[2], 1],
                         strides=[1, 1, 1],
                         activation=tf.nn.relu)  # (-1, 1, 1, 5, f)

        related_feature_conv = tf.reshape(related_feature_conv, (-1, related_feature_conv.shape[3]*related_feature_conv.shape[4]))
        fc_input = tf.concat([related_feature_conv] + feature_list, axis=1)

        fc1 = tf.contrib.layers.fully_connected(fc_input, 1024, activation_fn=tf.nn.relu)
        fc1 = tf.nn.dropout(fc1, self.dropout_keep_prob)

        fc2 = tf.contrib.layers.fully_connected(fc1, 1024, activation_fn=tf.nn.relu)
        fc2 = tf.nn.dropout(fc2, self.dropout_keep_prob)

        fc3 = tf.contrib.layers.fully_connected(fc2, 1024, activation_fn=tf.nn.relu)
        fc3 = tf.nn.dropout(fc3, self.dropout_keep_prob)

        output = tf.contrib.layers.fully_connected(fc3, 333, activation_fn=tf.nn.relu)

        return output

class NMAE1(NN):
    pass

class NMAE2(NN):
    pass
