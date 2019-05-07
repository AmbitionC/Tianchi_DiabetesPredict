# coding:utf-8
#Author: chenhao
#date: Jan.24.2018
#Description: Tianchi Medical solution using Features (LGB)


import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')


train['性别'] = train['性别'].map({'男': 1, '女': 0, '??':0})
test['性别'] = test['性别'].map({'男': 1, '女': 0, '??':0})

train['体检日期'] = (pd.to_datetime(train['体检日期']) - parse('2017-9-10')).dt.days
test['体检日期'] = (pd.to_datetime(test['体检日期']) - parse('2017-9-10')).dt.days

#train_y = train['血糖']
#train_x = train.drop(['id','血糖', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)
test_x = test.drop(['id', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)

# 删除缺少比较较少的参数的行
train = train.drop(train.loc[train['红细胞计数'].isnull()].index)
train = train.drop(train.loc[train['红细胞平均体积'].isnull()].index)
train = train.drop(train.loc[train['红细胞平均血红蛋白浓度'].isnull()].index)
train = train.drop(train.loc[train['白细胞计数'].isnull()].index)

train.fillna(train.median(axis=0), inplace=True)
test_x.fillna(test_x.median(axis=0), inplace=True)

# 删除离群值
train = train.drop(train[train['*r-谷氨酰基转换酶'] > 600].index)
train = train.drop(train[train['白细胞计数'] > 20.06].index)
train = train.drop(train[train['*丙氨酸氨基转换酶'] == 498.89].index)
train = train.drop(train[train['单核细胞%'] > 20].index)
train = train.drop(train[train['*碱性磷酸酶'] > 340].index)  # 有待调整
train = train.drop(train[train['*球蛋白'] > 60].index)
train = train.drop(train[train['嗜酸细胞%'] > 20].index)
train = train.drop(train[train['*天门冬氨酸氨基转换酶'] > 300].index)
train = train.drop(train[train['血小板计数'] > 700].index)
train = train.drop(train[train['*总蛋白'] > 100].index)

train_y = train['血糖']
train_x = train.drop(['id','血糖', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)

shape = np.shape(train_x)
print('train:',shape)

array = np.array(train_x)
poly = PolynomialFeatures(2,interaction_only=True)
train_x_trans = poly.fit_transform(array)
shape = np.shape(train_x_trans)
print('train:',shape)

array1 = np.array(test_x)
poly = PolynomialFeatures(2,interaction_only=True)
test_x_trans = poly.fit_transform(array1)
shape1 = np.shape(test_x_trans)
print('test:',shape1)

train_out = xgb.DMatrix(train_x_trans, label=train_y)
test_out = xgb.DMatrix(test_x_trans)

test_preds = np.zeros((test_x_trans.shape[0],1))
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
