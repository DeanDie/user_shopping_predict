# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:15:08 2018
@author: mokun
"""

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

import scipy as sp
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier


from data_processing import one_hot
from merge import merge, agg_table


OFF_LINE = False

categorical_feature = []

def xgb_model(train_set_x,train_set_y,test_set_x, params={}):
    # train_set_x = one_hot(train_set_x, categorical_feature)
    # test_set_x = one_hot(test_set_x, categorical_feature)

    # 模型参数
    if params == {}:
        params = {'booster': 'gbtree',
                  'objective':'binary:logistic',
                  'eta': 0.01,
                  'max_depth': 6,  # 4 3
                  'colsample_bytree': 0.8,#0.8
                  'subsample': 0.9,
                  'min_child_weight': 9,  # 2 3
                  'silent':1,
                  'nthread': 6,
                  'eval_metric': 'auc',
                  'seed': 2018,
                  }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=2000)
    predict = model.predict(dvali, ntree_limit=model.best_iteration)
    return predict



def xgb_valid(train_x,train_y,test_x):
    # train_x = one_hot(train_x)
    # test_x = one_hot(test_x)

    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              # 'eta': 0.01,
            #   'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              # 'min_child_weight': 9,  # 2 3
              'silent':1,
              # 'nthread': 6,
              'eval_metric': 'auc',
              'seed': 2018
              }
    params_cv = {
        'max_depth':list(range(3,9,1)),
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'min_child_weight':list(range(2,10,1)),
    }

    # dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    # dvali = xgb.DMatrix(test_set_x)
    # model = xgb.train(params, dtrain, num_boost_round=1000)

    print('Train CV Started ...')
    estimator = xgb.XGBClassifier(**params)
    grid = GridSearchCV(estimator, param_grid=params_cv, scoring='roc_auc', n_jobs=6, iid=False, cv=7)

    grid.fit(train_x, train_y)
    print(grid.grid_scores_, grid.best_params_, grid.best_score_)
    params = {**params, **grid.best_params_}
    predict = xgb_model(train_x, train_y, test_x, params)
    return predict



def lgb_model(train_x, train_y, test_x):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 32,
        'learning_rate': 0.02,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'seed':2018,
    }
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=99)

    lgb_train = lgb.Dataset(train_x, train_y, feature_name=list(train_x.columns), categorical_feature=categorical_feature)
    lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train, feature_name=list(train_x.columns), categorical_feature=categorical_feature)
    test_x = lgb.Dataset(test_x, feature_name=list(train_x.columns), categorical_feature=categorical_feature)

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    verbose_eval=250,
                    early_stopping_rounds=40)

    pred_value = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
    auc = roc_auc_score(valid_y, pred_value)
    print('auc value:', auc)

    y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    return y_pred


def drop(data, col_to_drop=[]):
    # data['V9_V10'] = data['V9'] * data['V10']
    # if col_to_drop == []:
    #     corr = data.corr()
    #     for col in data.columns:
    #         if data.iloc[col, 'FLAG'] < 0.01:
    #             col_to_drop.append(col)
    # else:
    col_to_drop = ['V3', 'V14', 'V21', 'APP'] #
    data = data.drop(col_to_drop, axis=1)
    return data

def get_categorical_feature(train):
    for col in train.columns:
        # print("%s: %d" % (col, len(train[col].unique())))
        if len(train[col].unique()) < 10 and col != 'FLAG':
            categorical_feature.append(col)
    print('categorical feature:', categorical_feature)



if __name__ == '__main__':
    # main()

    m = True
    if m:

        train, test = merge()
        # train = log_table_(train_log, train)
        # train = pd.merge(train, merged, on=['USRID'], how='left')
        train.fillna(0,inplace=True)
        train = agg_table(train)
        train = drop(train)

        # get_categorical_feature(train)

        if OFF_LINE == True:
            train_x = train.drop(['USRID', 'FLAG'], axis=1).values
            train_y = train['FLAG'].values
            auc_list = []

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
            for train_index, test_index in skf.split(train_x, train_y):
                print('Train: %s | test: %s' % (train_index, test_index))
                X_train, X_test = train_x[train_index], train_x[test_index]
                y_train, y_test = train_y[train_index], train_y[test_index]

                pred_value = lgb_model(X_train, y_train, X_test)    ######################################
                # pred_value = lgb_model(X_train, y_train, X_test, y_test)    ###############################

                # print(pred_value)
                # print(y_test)

                pred_value = np.array(pred_value)
                pred_value = [ele + 1 for ele in pred_value]

                y_test = np.array(y_test)
                y_test = [ele + 1 for ele in y_test]

                fpr, tpr, thresholds = roc_curve(y_test, pred_value, pos_label=2)

                auc = metrics.auc(fpr, tpr)
                print('auc value:',auc)
                auc_list.append(auc)

            print('validate result:',np.mean(auc_list))
            sys.exit(32)


        test.fillna(0,inplace=True)
        test = agg_table(test)
        test = drop(test)

        # train and predict ...
        train_y = train['FLAG'].values
        train_x = train.drop(['USRID', 'FLAG'], axis=1).values
        test_x = test.drop(['USRID'], axis=1).values
        
        pred_result = xgb_model(train_x, train_y, test_x)
        print(pred_result.shape)

        sub = pd.DataFrame()
        sub['USRID'] = test['USRID']
        sub['target'] = pred_result

        sub.to_csv('./submit/%s.csv'%str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),index=None,sep='\t')