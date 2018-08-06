import time
import sys
from datetime import datetime
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
try:
    from pyfm.pylibfm import FM
except ImportError:
    from pyfm import fm
import pywFM

from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import SMOTE

import main
from main import drop, log_tabel
from data_processing import smote



def ont_hot(data):
    # col = data.columns
    data = smote()
    # print(data.columns)
    # data.columns = col

    categorical_feature = []
    for col in data.columns:
        print("%s: %d" % (col, len(data[col].unique())))
        if len(data[col].unique()) <= 10:
            categorical_feature.append(col)
    print('categorical feature:', categorical_feature)
    exit()
    data = pd.get_dummies(data, columns=categorical_feature)
    data = data.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    train_x = data[data['FLAG'] != -1]
    test_x = data[data['FLAG'] == -1]
    train_y = data[data['FLAG'] != -1]['FLAG'].values
    # print(train_y.shape)
    # m = train_y.shape[0]

    # z-score
    # label = data.pop('FLAG')
    # print(label.unique())

    # data = pd.concat([data, label], axis=1)
    # data = data.to_dict('records')

    # v = DictVectorizer()
    # v.fit(data)
    # train_x, test_x = data[:m], data[m:]
    # train_x = v.transform(train_x)
    # test_x = v.transform(test_x)

    # train = data[data['FLAG'] >= 0]
    # test = data[data['FLAG'] < 0]
    # train_y = train['FLAG'].values
    # print('label:', (data['FLAG'].unique()))
    #
    # train_x = train.drop(['USRID'], axis=1)
    # test_x = test.drop(['USRID'], axis=1)
    #
    # train_x = train_x.to_dict('records')
    # test_x = test_x.to_dict('records')

    # exit()


    return train_x, train_y, test_x

train_agg = pd.read_csv('./train/train_agg.csv', sep='\t')
test_agg = pd.read_csv('./test/test_agg.csv', sep='\t')
agg = pd.concat([train_agg, test_agg], copy=False)
# print(train_agg.shape, test_agg.shape)

# 日志信息
train_log = pd.read_csv('./train/train_log.csv', sep='\t', parse_dates = ['OCC_TIM'])
test_log = pd.read_csv('./test/test_log.csv', sep='\t', parse_dates = ['OCC_TIM'])
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
data.fillna(0,inplace=True)

train_x, train_y, test_x = ont_hot(data)
print('##########################')
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

# train = data[data['FLAG'] >= 0]
#
# test = data[data['FLAG'] < 0]
# train_y = train['FLAG'].values

# fm = FM(num_factors=10, num_iter=300, verbose=True, task='classification', initial_learning_rate=0.01, learning_rate_schedule="optimal")
# fm.fit(train_x, train_y)
# y_pred = fm.predict(test_x)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018)
fm = pywFM.FM(task='classification', num_iter=100, k2=10, verbose=True)
test_y = np.ones(test_x.shape[0])
fm.run(train_x,train_y, test_x, test_y, valid_x, valid_y)
y_pred = fm.predictions
# print(roc_auc_score(valid_y, ))

sub = pd.DataFrame()
sub['USRID'] = test_x['USRID']
sub['target'] = y_pred

sub.to_csv('./submit/%s.csv'%str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),index=None,sep='\t')

# print('train set:', roc_auc_score(train_y, fm.predict(train_x)))

# help(pylibfm.FM)

