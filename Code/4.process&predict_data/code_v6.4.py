# coding=utf-8
'''
Author: chenhao
date: Jan.25.2018
Description: Use the train_Drop_Delete_Log.csv and PolynomialFeature in XGB model
'''
import pandas as pd
import datetime
import numpy as np
from dateutil.parser import parse

modelfile = 'modelweight.model' #神经网络权重保存

data_path = 'data/'

train = pd.read_csv(data_path + 'train_Drop_Delete_Log_Poly_keras.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'test_Drop_Delete_Log_Poly_keras.csv', encoding='gb2312')

# 对数据简单处理
train_y = train['BS']
train_x = train.drop(['BS'],axis=1)
test_x = test

test_preds = np.zeros((test_x.shape[0],1))

train_x = train_x.as_matrix()
train_y = train_y.as_matrix()
test_x = test_x.as_matrix()

# 3 建立一个简单BP神经网络模型
from keras.models import Sequential
from keras.layers.core import Dense, Activation
model = Sequential()  #层次模型
model.add(Dense(100,input_dim=801,init='uniform')) #输入层，Dense表示BP层
model.add(Activation('relu'))  #添加激活函数
model.add(Dense(1,input_dim=100))  #输出层
model.compile(loss='mean_squared_error', optimizer='adam') #编译模型
model.fit(train_x, train_y, nb_epoch = 1000, batch_size = 6) #训练模型1000次
model.save_weights(modelfile) #保存模型权重

#4 预测，并还原结果。
#x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
#data[u'L1_pred'] = model.predict(x) * data_std['L1'] + data_mean['L1']


test_preds = model.predict(test_x)
print(test_preds)

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False, float_format='%.3f')

