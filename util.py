import pandas as pd
import numpy as np
import os


def num_score(n, m, s=2, B=1):
    """
    n: predict value, m: real value
    s: Scale Factor (for each column), B: Scale Factor"""
    score = B * np.exp(-((n - m) / s) ** 2)
    return score

def cat_score(c, d, C=1):
    """
    c: predict values, d: real values, C: Scale Factor"""
    score = C * (c == d)
    return score


def calc_metric(label, pred, nan_position, etler):
    val_imputed_df = pd.DataFrame(np.array(pred), columns=etler.test_input_df.columns)
    val_imputed_df = etler.generate_output_df(val_imputed_df)

    original_df = pd.DataFrame(np.array(label), columns=etler.test_input_df.columns)
    original_df = etler.generate_output_df(original_df)

    columns = val_imputed_df.columns

    correct_dict = dict.fromkeys(columns, 0)
    score_dict = dict.fromkeys(columns, 0)

    nan_position = nan_position[:, [5, 6, 0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    nan_list = (len(nan_position) - nan_position.sum(axis=0)).tolist()
    for enum, col in enumerate(columns):
        # numeric score
        if col in columns[2:7]:
            for idx in range(len(val_imputed_df)):
                if nan_position[idx][enum] == 0:
                    pred = float(val_imputed_df[col].iloc[idx])
                    y = float(original_df[col].iloc[idx])
                    if y == round(pred):
                        correct_dict[col] += 1
                        score_dict[col] += num_score(pred, y)

        # categorical score
        else:
            for idx in range(len(val_imputed_df)):
                if nan_position[idx][enum] == 0:
                    pred = val_imputed_df[col].iloc[idx]
                    y = original_df[col].iloc[idx]
                    if y == pred:
                        correct_dict[col] += 1
                        score_dict[col] += cat_score(pred, y)
    df_metric = pd.DataFrame()
    df_metric["acc"] = pd.Series(correct_dict) / pd.Series(nan_list, index=columns)
    df_metric["score"] = pd.Series(score_dict) / pd.Series(nan_list, index=columns)
    df_metric["correct"] = pd.Series(correct_dict)
    df_metric["total"] = pd.Series(nan_list, index=columns)

    return df_metric['acc'], df_metric['score']

def fill_result_df(result_df, answer_df):
    """
    :params
        result_df: result_kor.csv 와 같은 형식의 DataFrame
        answer_df: test_kor.csv 와 같은 형식에, nan이 모두 채워진 DataFrame

    :return
        result_df와 같은 형식에, '값'column이 answer_df에 저장된 값으로 채워진 DataFrame
    """

    def _get_value(row):
        i = row['row'] - 2
        j = ord(row['column']) - ord('A')
        return (*row[:-1], answer_df.iat[i, j])

    return result_df.apply(_get_value, axis=1, result_type='broadcast')


def pick_final_result(epoch):
    result_df = pd.read_csv(os.path.join('result', 'result_{}epoch.csv'.format(epoch)), encoding='cp949')
    result_df.to_csv('result.csv', encoding='cp949', index=False)
