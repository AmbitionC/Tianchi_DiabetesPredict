# coding:utf-8
'''
#Author: chenhao
#date: Jan.22.2018
#Description: Tianchi Medical solution train dataset with Lightgbm, use the coxbox to soft the dataset
'''

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from scipy import stats
'''
############################################################
#模型融合
############################################################
data_path = 'data/'

lgb = pd.read_csv(data_path + 'sub20180129_202115.csv', encoding='gb2312')
xgb = pd.read_csv(data_path + 'sub20180129_202635.csv', encoding='gb2312')
catb = pd.read_csv(data_path + 'sub20180129_204109.csv', encoding='gb2312')
keras = pd.read_csv(data_path + 'sub20180129_215049.csv', encoding='gb2312')

keras = keras.as_matrix()
lgb = lgb.as_matrix()
xgb = xgb.as_matrix()
catb = catb.as_matrix()

#out = 0.5*keras
out = 0.37*keras + 0.21*lgb + 0.21*xgb + 0.21*catb
#print(out)

submission = pd.DataFrame({'pred': out.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,index=False, float_format='%.3f')
'''


############################################################
#高低血糖梯度融合法
############################################################
data_path = 'data/'

keras = pd.read_csv(data_path + 'sub20180129_220159.csv', encoding='gb2312')
lgb = pd.read_csv(data_path + 'sub20180129_220411.csv', encoding='gb2312')
xgb = pd.read_csv(data_path + 'sub20180129_220809.csv', encoding='gb2312')

keras = keras.as_matrix()
lgb = lgb.as_matrix()
xgb = xgb.as_matrix()

#out = 0.5*keras
out = 0.6*keras + 0.2*lgb + 0.2*xgb
#print(out)

submission = pd.DataFrame({'pred': out.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,index=False, float_format='%.3f')
