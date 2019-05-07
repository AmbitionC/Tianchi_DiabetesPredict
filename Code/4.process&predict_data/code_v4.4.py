# coding=utf-8
# Author: chenhao
# date: Jan.24.2018
# Description: Tianchi Medical solution using stacking (keras)
import pandas as pd
import datetime
import numpy as np
from dateutil.parser import parse

modelfile = 'modelweight.model' #神经网络权重保存

data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_chenhao.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_chenhao.csv', encoding='gb2312')


train['date'] = (pd.to_datetime(train['date']) - parse('2017-09-10')).dt.days
test['date'] = (pd.to_datetime(test['date']) - parse('2017-09-10')).dt.days

#删除缺少比较较少的参数的行
train = train.drop(train.loc[train['V'].isnull()].index)
train = train.drop(train.loc[train['Y'].isnull()].index)
train = train.drop(train.loc[train['AA'].isnull()].index)
train = train.drop(train.loc[train['U'].isnull()].index)

train.fillna(train.median(axis=0), inplace=True)
test.fillna(test.median(axis=0), inplace=True)



# 删除离群值
train = train.drop(train[train['D'] > 600].index)
train = train.drop(train[train['U'] > 20.06].index)
train = train.drop(train[train['B'] == 498.89].index)
train = train.drop(train[train['II'] > 20].index)
train = train.drop(train[train['C'] > 340].index)  # 有待调整
train = train.drop(train[train['G'] > 60].index)
train = train.drop(train[train['JJ'] > 20].index)
train = train.drop(train[train['A'] > 300].index)
train = train.drop(train[train['CC'] > 700].index)
train = train.drop(train[train['E'] > 100].index)

# 对数据简单处理
train_y = train['BS']
train_x = train.drop(['id','BS','P','Q','R','S','T'],axis=1)
test_x = test.drop(['id','P','Q','R','S','T'],axis=1)

# 对数据简单处理
#train_y = train['BS']
#train_x = train.drop(['id','BS'],axis=1)
#test_x = test.drop(['id'],axis=1)


# 中位数填充法填充缺失数据
train_x.fillna(train_x.median(axis=0), inplace=True)
train_y.fillna(train_y.median(axis=0), inplace=True)
test_x.fillna(test_x.median(axis=0), inplace=True)

#train_x = np.array(train_x)
#train_y = np.array(train_y)
#test_x = np.array(test_x)

test_preds = np.zeros((train_x.shape[0],1))

#feature = ['sex','age','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','BB','CC','DD','EE','FF','GG','HH','II','JJ','KK']
#label = ['BS']
train_x = train_x.as_matrix()
train_y = train_y.as_matrix()
#test_x = train_x.as_matrix()

# 3 建立一个简单BP神经网络模型
from keras.models import Sequential
from keras.layers.core import Dense, Activation
model = Sequential()  #层次模型
model.add(Dense(12,input_dim=35,init='uniform')) #输入层，Dense表示BP层
model.add(Activation('relu'))  #添加激活函数
model.add(Dense(1,input_dim=12))  #输出层
model.compile(loss='mean_squared_error', optimizer='adam') #编译模型
model.fit(train_x, train_y, nb_epoch = 1000, batch_size = 6) #训练模型1000次
model.save_weights(modelfile) #保存模型权重

#4 预测，并还原结果。
#x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
#data[u'L1_pred'] = model.predict(x) * data_std['L1'] + data_mean['L1']

#test_preds = model.predict(test_x)
#print test_preds

test_preds = model.predict(train_x)
print(test_preds)

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False, float_format='%.3f')

