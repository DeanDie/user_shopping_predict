import time
import sys
from datetime import datetime
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import SMOTE
from collections import Counter

# from main import drop, log_tabel



def read():
    train_agg = pd.read_csv('./train/train_agg.csv', sep='\t')
    test_agg = pd.read_csv('./test/test_agg.csv', sep='\t')
    agg = pd.concat([train_agg, test_agg], copy=False)
    print(train_agg.shape, test_agg.shape)

    # 日志信息
    train_log = pd.read_csv('./train/train_log.csv', sep='\t', parse_dates=['OCC_TIM'])
    test_log = pd.read_csv('./test/test_log.csv', sep='\t', parse_dates=['OCC_TIM'])
    log = pd.concat([train_log, test_log], copy=False)

    # log['EVT_LBL_1'] = log['EVT_LBL'].apply(lambda x: x.split('-')[0])
    # log['EVT_LBL_2'] = log['EVT_LBL'].apply(lambda x: x.split('-')[1])
    # log['EVT_LBL_3'] = log['EVT_LBL'].apply(lambda x: x.split('-')[2])

    # 用户唯一标识
    train_flg = pd.read_csv('./train/train_flg.csv', sep='\t')
    test_flg = pd.DataFrame()
    test_flg['USRID'] = test_agg['USRID']
    test_flg['FLAG'] = -1
    # del test_flg['RST']
    flg = pd.concat([train_flg, test_flg], copy=False)

    data = pd.merge(agg, flg, on=['USRID'], how='left', copy=False)

    # log['OCC_TIM'] = log['OCC_TIM'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    # log = log.sort_values(['USRID', 'OCC_TIM'])
    # log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)

    # merged = log_tabel(log)
    # data = pd.merge(data, merged, on=['USRID'], how='left')
    # data = drop(data)
    data.fillna(0, inplace=True)
    return data


def smote(data=None):
    if data == None:
        data = read()

    left = data[data['FLAG'] == -1]
    data = data[data['FLAG'] != -1]

    y = data.pop('FLAG')
    # left.drop(['FLAG'], inplace=True, axis=1)
    # print(left.columns)

    sm = SMOTE(random_state=99)
    x, y = sm.fit_sample(data, y)
    # print('Resampled dataset shape {}'.format(Counter(y)))
    data = pd.DataFrame(x)
    data.columns = [col for col in list(left.columns) if col != 'FLAG']
    data['FLAG'] = y

    # print('data shape:', data.shape)
    # print('left shape:', left.shape)
    data = pd.concat([data, left])
    # print('after merged, data shape:', data.shape)
    return data


def one_hot(data, categorical_feature=[]):
    # col = data.columns
    # data = smote()
    # print(data.columns)
    # data.columns = col

    if categorical_feature == []:
        for col in data.columns:
            # print("%s: %d" % (col, len(data[col].unique())))
            if len(data[col].unique()) <= 10:
                categorical_feature.append(col)
    # print('categorical feature:', categorical_feature)
    data = pd.get_dummies(data, columns=categorical_feature)
