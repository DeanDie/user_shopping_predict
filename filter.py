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


from data_processing import one_hot


def filtration(corr, relevant_threshold=0.85, threshold=0.1):
    # if
    conbination = [col for col in corr.columns if col.startswith('V') and len(col.split('_')) > 1]

    to_drop = []
    for col in conbination:
        x, y, _ = col.split('_')
        # print(x, y)
        if corr.loc[x, col] > relevant_threshold or corr.loc[y, col] > relevant_threshold:
            to_drop.append(col)
        if corr.loc[col, 'FLAG'] <= threshold:
            to_drop.append(col)

    print('To drop:', len(to_drop))

    return to_drop


def read_corr():
    corr = pd.read_csv('./train/corr.csv', index_col=0)

    col_to_drop = filtration(corr)


if __name__ == '__main__':
    read_corr()