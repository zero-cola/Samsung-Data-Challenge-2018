import numpy as np
import pandas as pd
import os

from constants import NP_SEED, DATA_DIR, DATA_FILE, PROBLEM_FILE, RESULT_FILE
np.random.seed(NP_SEED)

class ETL:
    """
    data_df: 학습, 검증에 사용할 수 있는 데이터
    problem_df: 삼성에서 문제로 주어진 데이터 (Nan이 뚫려있음)
    result_df:  결과 제출 양식. 위 두 df와는 다른 형태임.
    """
    def __init__(self):
        self.data_df = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE), encoding='cp949')
        self.problem_df = pd.read_csv(os.path.join(DATA_DIR, PROBLEM_FILE), encoding='cp949')
        self.result_df = pd.read_csv(os.path.join(DATA_DIR, RESULT_FILE), encoding='cp949')

        self.data_df = self.data_df[self.problem_df.columns]
        self.num_vars = ['사망자수', '사상자수', '중상자수', '경상자수', '부상신고자수']
        self.cat_vars = [col for col in self.problem_df.columns if col not in self.num_vars]
        print('num_vars:', self.num_vars, '\ncat_vars:', self.cat_vars, end='\n\n')
        print('# of Columns   :', len(self.data_df.columns), '=', len(self.num_vars), '+', len(self.cat_vars))
        print('# of instances :', len(self.data_df))

        # generate_input_df 함수가 초기화 진행함
        # data_df과 problem_df를 NMAE에 넣을 수 있도록 포맷팅한 df들.
        self.input_df = None
        self.train_input_df = None
        self.val_input_df = None
        self.test_input_df = None
        self.problem_input_df = None

        self.val_nan_mask = None
        self.test_nan_mask = None
        self.val_nan_pos = None
        self.test_nan_pos = None

        # self.col_attr_list: column과 그에 따른 attribute list
        # e.g. [['주야:야간', '주야:주간'],
        #       ['요일:금', '요일:목', '요일:수', '요일:월', '요일:일', '요일:토', '요일:화'], ...]
        self.col_attr_list = None
        self.attr_num_list = None  # e.g. [1, 1, 1, 1, 1, 2, 7, 17, 208, 4, 19, 20, 9, 16, 12, 14]

    def generate_input_df(self, drop_num_cols, drop_cat_cols, split_ratio=(0.7, 0.15, 0.15)):
        """
        수치형 변수(5개)와 범주형 변수(11개)가 섞여있는 원래 데이터를,
        수치형 변수는 그대로 사용하고, 범주형 변수를 각각 one-hot encoding하여
        data_df.shape == (-1, 16) -> input_df.shape == (-1, 333)으로 포매팅한다.

        그 후,
        1. 데이터 셔플
        2. train, val, test split

        :param split_ratio: (Train, Validation, Test)
        :return: data_df와 problem_df를 concat한 뒤 포매팅한 DataFrame
        """
        self.input_df = self.data_df.copy()

        self.input_df = pd.concat([self.input_df, self.problem_df], axis=0, ignore_index=True)
        print('\n # of test instances:', len(self.problem_df))
        print('test instance indices: {} ~ {}'.format(len(self.data_df), len(self.input_df) - 1))

        self.input_df, self.col_attr_list, self.attr_num_list = self.onehot_encoding(self.input_df)

        val_split_idx = int(len(self.data_df) * split_ratio[0])
        test_split_idx = val_split_idx + int(len(self.data_df) * split_ratio[1])

        # Data Random Shuffle
        p = np.random.permutation(len(self.data_df))
        data_input_df = self.input_df.iloc[p]

        self.train_input_df = data_input_df.iloc[: val_split_idx]
        self.val_input_df = data_input_df.iloc[val_split_idx: test_split_idx]
        self.test_input_df = data_input_df.iloc[test_split_idx:]
        self.problem_input_df = self.input_df.iloc[-len(self.problem_df):]

        self.val_nan_mask, self.val_nan_pos = self.gen_random_nan_mask(len(self.val_input_df), drop_num_cols, drop_cat_cols)
        self.test_nan_mask, self.test_nan_pos = self.gen_random_nan_mask(len(self.test_input_df), drop_num_cols, drop_cat_cols)

        print('train: {}, val: {}, test: {}'.format(
                    len(self.train_input_df), len(self.val_input_df), len(self.test_input_df)))
        print('problem: {}'.format(len(self.problem_input_df)))

        return self.input_df

    def onehot_encoding(self, data_df):
        def col_attr_name(col: str, attr: list):
            """
            col: column 이름
            attr: attribure 이름 list
            """
            col_attr = []
            for att in attr:
                col_attr.append(col + ':' + att)
            return col_attr

        cat_df = data_df[self.cat_vars]
        data_df.drop(self.cat_vars, axis=1, inplace=True)

        df_list = [data_df]
        cat_column_list = []
        attr_num_list = [1, 1, 1, 1, 1]


        for column in cat_df.columns:
            na_temp = cat_df[column].isnull()
            temp = pd.get_dummies(cat_df[column])
            temp[na_temp] = np.nan

            col_attr_dict = {k: v for k, v in zip(temp.columns.tolist(),
                                                  col_attr_name(column, temp.columns.tolist()))}
            temp = temp.rename(columns=col_attr_dict)
            df_list.append(temp)
            cat_column_list.append(list(temp.columns.values))
            attr_num_list.append(len(temp.columns))

        data_df = pd.concat(df_list, axis=1)
        return data_df, cat_column_list, attr_num_list

    def sparse_encoding(self, data_df):
        num_df = data_df[self.num_vars]
        data_df.drop(columns=self.num_vars, inplace=True)

        cat_series_list = []
        for col in self.col_attr_list:
            cat_series = data_df[col].idxmax(axis=1).apply(lambda x: x.split(':')[-1])
            cat_series.name = col[0].split(':')[0]
            cat_series_list.append(cat_series)

        data_df = pd.concat([num_df]+cat_series_list, axis=1)
        data_df = data_df[self.problem_df.columns]

        return data_df

    def generate_output_df(self, output_df):
        output_df = self.sparse_encoding(output_df)
        return output_df

    def gen_random_nan_mask(self, batch_size, drop_num_cols=3, drop_cat_cols=3):
        """
        수치형 변수와 범주형 변수에서 각각 drop_num_cols, drop_cat_cols개 만큼 빈칸을 생성한다.
        만약 -1로 설정 시, 2 또는 3개의 빈칸이 뚫린다.
        함수의 용도는 위와 같지만 실제로는 boolean mask를 생성하는 것.
        nan -> 0
        else -> 1

        return: boolmasks(-1, 333), nan_positions(-1, 16)
        """
        boolmasks = []
        nan_positions = []
        for i in range(batch_size):
            if drop_cat_cols == -1:
                selected_cols = np.random.choice(range(5, 16), np.random.randint(2, 4), replace=False)
            else:
                selected_cols = np.random.choice(range(5, 16), drop_cat_cols, replace=False)
            boolmask = []
            nan_position = []
            for j in range(len(self.attr_num_list)):
                n_attr = self.attr_num_list[j]
                if j in selected_cols:
                    nan_position[len(nan_position):] = [0]
                    boolmask[len(boolmask):] = [0] * n_attr
                else:
                    nan_position[len(nan_position):] = [1]
                    boolmask[len(boolmask):] = [1] * n_attr
            boolmasks.append(boolmask)
            nan_positions.append(nan_position)

        boolmasks = np.array(boolmasks)
        nan_positions = np.array(nan_positions)
        if drop_num_cols == -1:
            for boolmask, nan_position in zip(boolmasks, nan_positions):
                selected_num_cols = np.random.choice(range(5), np.random.randint(2, 4), replace=False)
                boolmask[selected_num_cols] = 0
                nan_position[selected_num_cols] = 0

        else:
            for boolmask, nan_position in zip(boolmasks, nan_positions):
                selected_num_cols = np.random.choice(range(5), np.random.randint(2, 4), replace=False)
                boolmask[selected_num_cols] = 0
                nan_position[selected_num_cols] = 0

        return boolmasks, nan_positions

    def gen_problem_nan_mask(self, batch_size):
        """
        problem에 빈칸 뚫린 방식들 중 batch_size개를 골라 nan mask를 만드는 함수

        :param problem_df: ETL.problem_df. Nan이 존재해야 함.
        :param problem_input_df: ETL.problem_input_df. Nan이 존재해야 함.
        :return: boolmasks(-1, 333), nan_positions(-1, 16)
        """
        problem_boolmasks = self.problem_input_df.notnull().values.astype(float)
        problem_nan_positions = self.problem_df.notnull().values.astype(float)
        choices = np.random.choice(range(len(problem_boolmasks)), batch_size)
        boolmasks = problem_boolmasks[choices]
        nan_positions = problem_nan_positions[choices]

        return boolmasks, nan_positions