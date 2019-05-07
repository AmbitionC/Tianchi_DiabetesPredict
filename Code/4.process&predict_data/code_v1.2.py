# coding:utf-8
#Author: chenhao
#date: Jan.16.2018
#Description: Tianchi Medical solution train dataset with XGBoost

import time
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')



train_y = train['血糖']
train_x = train.drop(['id','体检日期','血糖'], axis=1)
test_x = test.drop(['id','体检日期'], axis=1)

train_x.fillna(train_x.median(axis=0), inplace=True)
test_x.fillna(test_x.median(axis=0), inplace=True)

train_out = xgb.DMatrix(train_x, label=train_y)
test_out = xgb.DMatrix(test_x)

test_preds = np.zeros((test_x.shape[0],1))
#submission = pd.DataFrame({'label': test_preds.mean(axis=1)})
#submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False, float_format='%.3f')

#print (test_preds)

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.7,
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

