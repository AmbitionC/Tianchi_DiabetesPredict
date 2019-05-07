# coding:utf-8
#Author: chenhao
#date: Jan.22.2018
#Description: Tianchi Medical solution train dataset with Lightgbm, use the coxbox to soft the dataset

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from scipy import stats

data_path = 'data/'

keras = pd.read_csv(data_path + 'keras.csv', encoding='gb2312')
lgb = pd.read_csv(data_path + 'lgb.csv', encoding='gb2312')
xgb = pd.read_csv(data_path + 'xgb.csv', encoding='gb2312')

keras = keras.as_matrix()
lgb = lgb.as_matrix()
xgb = xgb.as_matrix()

#out = 0.5*keras
out = 0.34*keras + 0.33*lgb + 0.33*xgb
#print(out)

submission = pd.DataFrame({'pred': out.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,index=False, float_format='%.3f')