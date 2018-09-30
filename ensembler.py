import numpy as np
import pandas as pd
import util
from visualizer import plot_metrics
from constants import MODEL_DIR
import os
import tensorflow as tf

class Ensembler:

    def __init__(self, etler, batch_size, total_epoch, train_all_data):

        self.etler = etler
        self.batch_size = batch_size
        self.total_epoch = total_epoch
        self.train_all_data = train_all_data

    def train_per_nn(self, sess, nn_list):
        """
        nn_list에 있는 nn 각각을 total_epoch까지 학습한 뒤, 앙상블
        """

        ensemble_metrics = {'val': {'score': []},
                            'test': {'score': []}}

        # nn 각각 total_epoch까지 학습
        for nn in nn_list:
            nn.train(self.etler, -1, -1,
                     self.batch_size,
                     self.total_epoch,
                     self.train_all_data,  # val, test 데이터를 학습에 사용할 지 여부
                     )
        self.save_model(sess)

        print("\n[Ensembled Model Testing]")
        for i in range(self.total_epoch):
            print("[Ensemble EPOCH: {}]".format(i))
            val_pred = np.zeros((len(self.etler.val_input_df), 333))
            test_pred = np.zeros((len(self.etler.test_input_df), 333))
            problem_pred = np.zeros((len(self.etler.problem_input_df), 333))

            # 각각의 nn 모델의 softmax 결과값을 합산한뒤, argmax로 최종 예측값 도출
            for nn in nn_list:
                val_pred += nn.predicts['val']['x_pred'][i]
                test_pred += nn.predicts['test']['x_pred'][i]
                problem_pred += nn.predicts['problem']['x_pred'][i]

            val_pred = val_pred / len(nn_list)
            test_pred = test_pred / len(nn_list)
            problem_pred = problem_pred / len(nn_list)

            val_acc, val_score = util.calc_metric(self.etler.val_input_df.values, val_pred, self.etler.val_nan_pos, self.etler)
            test_acc, test_score = util.calc_metric(self.etler.test_input_df.values, test_pred, self.etler.test_nan_pos, self.etler)
            print('Validation Score')
            print(val_score)
            print('Test Score')
            print(test_score)

            print('[SUMMARY]')
            print('[val ] score: {:.4} num_score: {:.4} cat_score: {:.4}'.format(val_score.mean(),
                                                                          val_score[self.etler.num_vars].mean(),
                                                                          val_score[self.etler.cat_vars].mean()))
            print('[test] score: {:.4} num_score: {:.4} cat_score: {:.4}'.format(test_score.mean(),
                                                                          test_score[self.etler.num_vars].mean(),
                                                                          test_score[self.etler.cat_vars].mean()))

            ensemble_metrics['val']['score'].append(val_score.mean())
            ensemble_metrics['test']['score'].append(test_score.mean())
            plot_metrics(**ensemble_metrics)

            # epoch당 앙상블 모델의 예측 결과를 엑셀파일로 저장
            # problem 데이터 예측 결과를 format에 맞게 만들기
            problem_imputed_df = pd.DataFrame(np.array(problem_pred), columns=self.etler.problem_input_df.columns)
            problem_imputed_df = self.etler.generate_output_df(problem_imputed_df)
            result_df = util.fill_result_df(self.etler.result_df, problem_imputed_df)

            # 정답지를 result/result_#epoch.csv 에 저장
            if not os.path.exists('result'):
                os.mkdir('result')
            result_df.to_csv(os.path.join('result', 'ensemble_{}epoch.csv'.format(i)), index=False, encoding='cp949')

    def restore_and_predict(self, sess, nn_list):
        problem_input_df = self.etler.problem_input_df
        problem_nan_mask = problem_input_df.notnull().values.astype(float)  # nan -> 0, else -> 1
        problem_input_df = problem_input_df.fillna(0)
        problem_data = problem_input_df.values

        result_df = self.etler.result_df

        self.restore_model(sess)
        problem_preds = np.zeros((len(self.etler.problem_input_df), 333))
        for nn in nn_list:
            problem_pred = nn.problem_predict(problem_data, problem_nan_mask)
            problem_preds += problem_pred

        problem_pred = problem_preds / len(nn_list)
        # problem 데이터 예측 결과를 format에 맞게 만들기
        problem_imputed_df = pd.DataFrame(np.array(problem_pred), columns=problem_input_df.columns)
        problem_imputed_df = self.etler.generate_output_df(problem_imputed_df)
        result_df = util.fill_result_df(result_df, problem_imputed_df)

        # 정답지를 result/result_#epoch.csv 에 저장
        if not os.path.exists('result'):
            os.mkdir('result')
        result_df.to_csv(os.path.join('result', 'ensemble_final.csv'), index=False, encoding='cp949')

    def save_model(self, sess):
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        saver = tf.train.Saver()
        save_path = os.path.join(MODEL_DIR, 'ensemble')
        saver.save(sess, save_path, write_meta_graph=False)

    def restore_model(self, sess):
        restore_path = os.path.join(MODEL_DIR, 'ensemble')
        meta_path = restore_path + '.meta'
        ckpt_path = restore_path
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)