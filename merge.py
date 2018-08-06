import time
import sys
from datetime import datetime
import gc
from collections import Counter
import os

import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score

import scipy as sp
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from scipy.stats import mode

from data_processing import one_hot



def get_max_accur(x):
    count = Counter(x)
#     print(count)
#     exit
    a = count[max(count,key=count.get)]
    return a


def log_table(data):
    data['day'] = data.OCC_TIM.map(lambda x: x.day)
    data['hour'] = data.OCC_TIM.map(lambda x: x.hour)
    data['minute'] = data.OCC_TIM.map(lambda x: x.minute)

    data['EVT_LBL_1'] = data['EVT_LBL'].apply(lambda x: x.split('-')[0])
    data['EVT_LBL_2'] = data['EVT_LBL'].apply(lambda x: x.split('-')[1])
    data['EVT_LBL_3'] = data['EVT_LBL'].apply(lambda x: x.split('-')[2])
    EVT_LBL_1 = data.groupby(by=['USRID', 'EVT_LBL_1'], as_index=False)['EVT_LBL_1'].agg({'EVT_LBL_1_len':len})
    EVT_LBL_1 = EVT_LBL_1.groupby(by=['USRID'], as_index=False)['EVT_LBL_1_len'].agg({
        'EVT_LBL_1_mean': np.mean,
        'EVT_LBL_1_std': np.std,
        'EVT_LBL_1_min': np.min,
        'EVT_LBL_1_max': np.max
    })
    EVT_LBL_1_max_accur = data.groupby(by=['USRID'], as_index=False)['EVT_LBL_1'].agg({'EVT_LBL_1_max_accur': lambda x: mode(x)[0][0]})

    EVT_LBL_2 = data.groupby(by=['USRID', 'EVT_LBL_2'], as_index=False)['EVT_LBL_2'].agg({'EVT_LBL_2_len': len})
    EVT_LBL_2 = EVT_LBL_2.groupby(by=['USRID'], as_index=False)['EVT_LBL_2_len'].agg({
        'EVT_LBL_2_mean': np.mean,
        'EVT_LBL_2_std': np.std,
        'EVT_LBL_2_min': np.min,
        'EVT_LBL_2_max': np.max
    })
    EVT_LBL_2_max_accur = data.groupby(by=['USRID'], as_index=False)['EVT_LBL_2'].agg(
        {'EVT_LBL_2_max_accur': lambda x: mode(x)[0][0]})

    EVT_LBL_3 = data.groupby(by=['USRID', 'EVT_LBL_3'], as_index=False)['EVT_LBL_3'].agg({'EVT_LBL_3_len': len})
    EVT_LBL_3 = EVT_LBL_3.groupby(by=['USRID'], as_index=False)['EVT_LBL_3_len'].agg({
        'EVT_LBL_3_mean': np.mean,
        'EVT_LBL_3_std': np.std,
        'EVT_LBL_3_min': np.min,
        'EVT_LBL_3_max': np.max
    })
    EVT_LBL_3_max_accur = data.groupby(by=['USRID'], as_index=False)['EVT_LBL_3'].agg(
        {'EVT_LBL_3_max_accur': lambda x: mode(x)[0][0]})

    EVT_LBL_len = data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg({'EVT_LBL_len': len})
    EVT_LBL_set_len = data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg(
        {'EVT_LBL_set_len': lambda x: len(set(x))})
    EVT_LBL_Max_Accurred = data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg({'max_accur': get_max_accur})

    data['OCC_TIM'] = data['OCC_TIM'].apply(lambda x: time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S")))
    data['next_time'] = data.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)
    next_time = data.groupby(['USRID'], as_index=False)['next_time'].agg({
        'next_time_mean': np.mean,
        'next_time_std': np.std,
        'next_time_min': np.min,
        'next_time_max': np.max
    })
    grouped = data.groupby(by=['USRID'], as_index=False)['TCH_TYP']
    APP_WEB_H5 = grouped.agg(
        {'APP': lambda x: Counter(x)[0], 'WEB': lambda x: Counter(x)[1], 'H5': lambda x: Counter(x)[2]})
    #     WEB = grouped.apply(lambda x: Counter(x)[1])
    #     H5 = grouped.apply(lambda x: Counter(x)[2])

    merged = EVT_LBL_len
    merged = pd.merge(merged, EVT_LBL_1, on=['USRID'], how='left')
    merged = pd.merge(merged, EVT_LBL_2, on=['USRID'], how='left')
    merged = pd.merge(merged, EVT_LBL_3, on=['USRID'], how='left')
    merged = pd.merge(merged, EVT_LBL_1_max_accur, on=['USRID'], how='left')
    merged = pd.merge(merged, EVT_LBL_2_max_accur, on=['USRID'], how='left')
    merged = pd.merge(merged, EVT_LBL_3_max_accur, on=['USRID'], how='left')

    merged = pd.merge(merged, EVT_LBL_set_len, on=['USRID'], how='left')
    merged = pd.merge(merged, EVT_LBL_Max_Accurred, on=['USRID'], how='left')

    merged = pd.merge(merged, next_time, on=['USRID'], how='left')
    merged = pd.merge(merged, APP_WEB_H5, on=['USRID'], how='left')

    return merged


def log_table_(data, merged):

    day = data.groupby(by=['USRID', 'day'], as_index=False)['day'].agg({'day_count': len})
    day = day.groupby(by=['USRID'], as_index=False)['day_count'].agg({
        'day_mean': np.mean,
        'day_std': np.std,
        'day_min': np.min,
        'day_max': np.max
    })

    hour = data.groupby(by=['USRID', 'day', 'hour'], as_index=False)['hour'].agg({'hour_count': len})
    hour = hour.groupby(by=['USRID'], as_index=False)['hour_count'].agg({
        'hour_mean': np.mean,
        'hour_std': np.std,
        'hour_min': np.min,
        'hour_max': np.max
    })

    minute = data.groupby(by=['USRID', 'minute'], as_index=False)['minute'].agg({'minute_count': len})
    minute = minute.groupby(by=['USRID'], as_index=False)['minute_count'].agg({
        'minute_mean': np.mean,
        'minute_std': np.std,
        'minute_min': np.min,
        'minute_max': np.max
    })

    merged = pd.merge(merged, day, on=['USRID'], how='left')
    merged = pd.merge(merged, hour, on=['USRID'], how='left')
    merged = pd.merge(merged, minute, on=['USRID'], how='left')
    return merged


def agg_table(data):
    cols = [col for col in data.columns if col.startswith('V')]
    for i in cols[:-1]:
        if i == 'FLAG':
            continue
        for j in cols[cols.index(i) + 1:]:
            if j == 'FLAG':
                continue
            col_name = i + '_' + j
            data[col_name + '_plus'] = data[i] + data[j]
            data[col_name + '_sub'] = data[i] - data[j]
            data[col_name + '_mul'] = data[i] * data[j]

    return data


def merge():
    # train
    train_agg = pd.read_csv('./train/train_agg.csv', sep='\t')
    train_flg = pd.read_csv('./train/train_flg.csv', sep='\t')
    train_log = pd.read_csv('./train/train_log.csv', sep='\t', parse_dates=['OCC_TIM'])
    train_log = train_log.sort_values(['USRID', 'OCC_TIM'])

    train = pd.merge(train_flg,train_agg,on=['USRID'],how='left')
    merged = log_table(train_log)
    merged = log_table_(train_log, merged)
    train = pd.merge(train, merged, on=['USRID'], how='left')

    # test
    test_agg = pd.read_csv('./test/test_agg.csv', sep='\t')
    test_log = pd.read_csv('./test/test_log.csv', sep='\t', parse_dates=['OCC_TIM'])
    test_log = test_log.sort_values(['USRID', 'OCC_TIM'])
    merged = log_table(test_log)
    merged = log_table_(test_log, merged)
    test = pd.merge(test_agg, merged, on=['USRID'], how='left')

    print('train shape:', train.shape)
    print('test shape:', test.shape)

    return train, test



if __name__ == '__main__':
    merge()