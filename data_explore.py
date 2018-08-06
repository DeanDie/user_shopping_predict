
import time
import sys
from datetime import datetime
import gc

import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

import scipy as sp
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier


from data_processing import one_hot
from merge import merge


# OFF_LINE = False
#
# categorical_feature = []
#
# def drop(data):
#     # data['V9_V10'] = data['V9'] * data['V10']
#     col_to_drop = ['V3', 'V14', 'V21', 'V9', 'WEB', 'APP'] #
#     data = data.drop(col_to_drop, axis=1)
#     return data
#
# def get_categorical_feature(train):
#     for col in train.columns:
#         # print("%s: %d" % (col, len(train[col].unique())))
#         if len(train[col].unique()) < 10 and col != 'FLAG':
#             categorical_feature.append(col)
#     print('categorical feature:', categorical_feature)



# train_agg = pd.read_csv('./train/train_agg.csv',sep='\t')
# train_flg = pd.read_csv('./train/train_flg.csv',sep='\t')
# train_log = pd.read_csv('./train/train_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
# train_log = train_log.sort_values(['USRID', 'OCC_TIM'])
#
# train, test = merge()
# # train = log_table_(train_log, train)
# # train = pd.merge(train, merged, on=['USRID'], how='left')
# train.fillna(0,inplace=True)
# train = drop(train)
#
# colormap = plt.cm.magma
# plt.figure(figsize=(20,14))
# plt.title('Pearson correlation of continuous features', y=1.05, size=15)
# sns.heatmap(train.corr(),linewidths=1,vmax=1.0, square=True,
#             cmap=colormap, linecolor='white', annot=True)
#
# plt.show()
import re
text = '人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e。/b 我/b  们/e 很/s 同/b 意/e。/s'

sentences = re.split('[，。！？、‘’“”]/[bems]', text)
print(sentences)
# Filter sentences whose length is 0
sentences = list(filter(lambda x: x.strip(), sentences))
print(sentences)
# Strip sentences
sentences = list(map(lambda x: x.strip(), sentences))
print(sentences)