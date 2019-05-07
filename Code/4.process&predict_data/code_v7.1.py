# coding=utf-8
'''
Author:chenhao
Date: Jan 25 ,2017
Description: Use the train_Drop_Delete_Log.csv and PolynomialFeature in XGB model
'''

import time
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

data_path = 'data/'

train = pd.read_csv(data_path + 'train_Drop_Delete_Log_Poly_Ratio_for_B.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'test_Drop_Delete_Log_Poly_Ratio_for_B.csv', encoding='gb2312')

train_y = train['血糖']
train_x = train.drop(['血糖'], axis=1)
test_x = test

train_out = xgb.DMatrix(train_x, label=train_y)
test_out = xgb.DMatrix(test_x)

test_preds = np.zeros((test_x.shape[0],1))

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.71,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.01,
          'tree_method': 'exact',
          'seed': 0,
          'nthread': 12
          }

watchlist = [(train_out,'train')]
model = xgb.train(params,train_out,num_boost_round=3000,evals=watchlist)

test_preds[:,0] = model.predict(test_out)
print(test_preds)


submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False, float_format='%.3f')

