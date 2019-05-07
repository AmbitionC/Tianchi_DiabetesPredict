# coding:utf-8
#Author: chenhao
#date: Jan.21.2018
#Description: Tianchi Medical solution train dataset with XGBoost,besides, use the coxbox to soft the dataset

import time
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from dateutil.parser import parse
from scipy import stats

data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')

train['性别'] = train['性别'].map({'男': 1, '女': 0})
test['性别'] = test['性别'].map({'男': 1, '女': 0})

train['体检日期'] = (pd.to_datetime(train['体检日期']) - parse('2017-9-10')).dt.days
test['体检日期'] = (pd.to_datetime(test['体检日期']) - parse('2017-9-10')).dt.days

#train_y = train['血糖']
#train_x = train.drop(['id','血糖'], axis=1)
#test_x = test.drop(['id'], axis=1)


#删除缺少比较较少的参数的行
#train = train.drop(train.loc[train['血红蛋白'].isnull()].index)
#train = train.drop(train.loc[train['红细胞平均血红蛋白浓度'].isnull()].index)
#train = train.drop(train.loc[train['白细胞计数'].isnull()].index)

train.fillna(train.median(axis=0), inplace=True)
test.fillna(test.median(axis=0), inplace=True)

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

#去除缺失值较多的数据
train_y = train['血糖']
train_x = train.drop(['id','血糖','乙肝表面抗原','乙肝表面抗体','乙肝核心抗体','乙肝e抗原','乙肝e抗体'], axis=1)
test_x = test.drop(['id','乙肝表面抗原','乙肝表面抗体','乙肝核心抗体','乙肝e抗原','乙肝e抗体'], axis=1)

'''
train_x['甘油三酯'] , a = stats.boxcox(train_x['甘油三酯'])
train_x['*r-谷氨酰基转换酶'] , b = stats.boxcox(train_x['*r-谷氨酰基转换酶'])
train_x['白球比例'] , c = stats.boxcox(train_x['白球比例'])
#train_x['乙肝e抗原'] , d = stats.boxcox(train_x['乙肝e抗原'])
train_x['*天门冬氨酸氨基转换酶'] , e = stats.boxcox(train_x['*天门冬氨酸氨基转换酶'])
train_x['*丙氨酸氨基转换酶'] , f = stats.boxcox(train_x['*丙氨酸氨基转换酶'])
#train_x['嗜酸细胞%'] , g = stats.boxcox(train_x['嗜酸细胞%'])
#train_x['乙肝核心抗体'] , h = stats.boxcox(train_x['乙肝核心抗体'])

test_x['甘油三酯'] , a1 = stats.boxcox(test_x['甘油三酯'])
test_x['*r-谷氨酰基转换酶'] , b1 = stats.boxcox(test_x['*r-谷氨酰基转换酶'])
test_x['白球比例'] , c1 = stats.boxcox(test_x['白球比例'])
#test_x['乙肝e抗原'] , d1 = stats.boxcox(test_x['乙肝e抗原'])
test_x['*天门冬氨酸氨基转换酶'] , e1 = stats.boxcox(test_x['*天门冬氨酸氨基转换酶'])
test_x['*丙氨酸氨基转换酶'] , f1 = stats.boxcox(test_x['*丙氨酸氨基转换酶'])
#test_x['嗜酸细胞%'] , g1 = stats.boxcox(test_x['嗜酸细胞%'])
#test_x['乙肝核心抗体'] , h1 = stats.boxcox(test_x['乙肝核心抗体'])
'''

train_out = xgb.DMatrix(train_x, label=train_y)
test_out = xgb.DMatrix(test_x)

test_preds = np.zeros((test_x.shape[0],1))
#submission = pd.DataFrame({'label': test_preds.mean(axis=1)})
#submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False, float_format='%.3f')

#print (test_preds)

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'mae',
          'gamma': 0.1,
          'min_child_weight': 1.3,
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

