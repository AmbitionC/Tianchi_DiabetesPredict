# coding=utf-8
# Author: chenhao
# date: Jan.17.2018
# Description: Tianchi Medical solution train dataset with keras
import pandas as pd
import datetime
import numpy as np
from dateutil.parser import parse

modelfile = 'modelweight.model' #神经网络权重保存

data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_chenhao.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_chenhao.csv', encoding='gb2312')

#获得最小的日期
#train['date'] = (pd.to_datetime(train['date']) - parse('2017-09-10')).dt.days
#b = pd.DataFrame(train['date'])
#a = b.describe()
#print a

train['date'] = (pd.to_datetime(train['date']) - parse('2017-09-10')).dt.days
test['date'] = (pd.to_datetime(test['date']) - parse('2017-09-10')).dt.days

# 对数据简单处理
train_y = train['BS']
train_x = train.drop(['id','BS'],axis=1)
test_x = test.drop(['id'],axis=1)

# 中位数填充法填充缺失数据
train_x.fillna(train_x.median(axis=0), inplace=True)
train_y.fillna(train_y.median(axis=0), inplace=True)
test_x.fillna(test_x.median(axis=0), inplace=True)

#train_x = np.array(train_x)
#train_y = np.array(train_y)
#test_x = np.array(test_x)

test_preds = np.zeros((test_x.shape[0],1))

#feature = ['sex','age','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','BB','CC','DD','EE','FF','GG','HH','II','JJ','KK']
#label = ['BS']
train_x = train_x.as_matrix()
train_y = train_y.as_matrix()
test_x = test_x.as_matrix()

# 3 建立一个简单BP神经网络模型
from keras.models import Sequential
from keras.layers.core import Dense, Activation
model = Sequential()  #层次模型
model.add(Dense(12,input_dim=40,init='uniform')) #输入层，Dense表示BP层
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

test_preds = model.predict(test_x)
print(test_preds)

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False, float_format='%.3f')






'''
#原始参考Keras代码
inputfile = 'input.xlsx'   #excel输入
outputfile = 'output.xls' #excel输出
modelfile = 'modelweight.model' #神经网络权重保存
data = pd.read_excel(inputfile,index='Date',sheetname=0) #pandas以DataFrame的格式读入excel表
feature = ['F1','F2','F3','F4'] #影响因素四个
label = ['L1'] #标签一个，即需要进行预测的值
data_train = data.loc[range(0,520)].copy() #标明excel表从第0行到520行是训练集

#2 数据预处理和标注
data_mean = data_train.mean()  
data_std = data_train.std()  
data_train = (data_train - data_mean)/data_std #数据标准化
x_train = data_train[feature].as_matrix() #特征数据
y_train = data_train[label].as_matrix() #标签数据

 #3 建立一个简单BP神经网络模型
from keras.models import Sequential
from keras.layers.core import Dense, Activation
model = Sequential()  #层次模型
model.add(Dense(12,input_dim=4,init='uniform')) #输入层，Dense表示BP层
model.add(Activation('relu'))  #添加激活函数
model.add(Dense(1,input_dim=12))  #输出层
model.compile(loss='mean_squared_error', optimizer='adma') #编译模型
model.fit(x_train, y_train, nb_epoch = 1000, batch_size = 6) #训练模型1000次
model.save_weights(modelfile) #保存模型权重

#4 预测，并还原结果。
x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
data[u'L1_pred'] = model.predict(x) * data_std['L1'] + data_mean['L1']

#5 导出结果
data.to_excel(outputfile) 

#6 画出预测结果图
import matplotlib.pyplot as plt 
p = data[['L1','L1_pred']].plot(subplots = True, style=['b-o','r-*'])
plt.show()

'''



'''
#Tensorflow参考代码
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io

# 调用数据
data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_chenhao.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_chenhao.csv', encoding='gb2312')


# 对数据简单处理
train_y = train['BS']
train_x = train.drop(['id','sex','date','BS'],axis=1)
test_x = test.drop(['id','date'],axis=1)

# 中位数填充法填充缺失数据
train_x.fillna(train_x.median(axis=0), inplace=True)
train_y.fillna(train_y.median(axis=0), inplace=True)
test_x.fillna(test_x.median(axis=0), inplace=True)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)

test_preds = np.zeros((test_x.shape[0],1))

input_dim = 38
X = tf.placeholder(tf.float32, [None,38,2])
Y = tf.placeholder(tf.float32, [None,1])

print train_x

#regression
def ass_rnn(hidden_layer_size=6):
    W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
    b = tf.Variable(tf.random_normal([1]), name='b')
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(X)[0], 1, 1])
    out = tf.matmul(outputs, W_repeated) + b
    out = tf.squeeze(out)
    return out


def train_rnn():
    out = ass_rnn()
    loss = tf.reduce_mean(tf.square(out - Y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        sess.run(tf.global_variables_initializer())

        for step in range(9000):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
            if step % 10 == 0:
                # 用测试数据评估loss
                print(step, loss_)
        print("保存模型: ", saver.save(sess, 'ass.model'))

def prediction():

    out = ass_rnn()

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        saver.restore(sess, './ass.model')

        prev_seq = train_x[-1]
        predict = []
        for i in range(12):
            next_seq = sess.run(out, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

train_rnn()

'''
'''
# 加载数据
url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/铁路客运量.csv'
ass_data = requests.get(url).content

df = pd.read_csv(io.StringIO(ass_data.decode('utf-8')))  # python2使用StringIO.StringIO

data = np.array(df['铁路客运量_当期值(万人)'])
# normalize
normalized_data = (data - np.mean(data)) / np.std(data)

seq_size = 3
train_x, train_y = [], []
for i in range(len(normalized_data) - seq_size - 1):
    train_x.append(np.expand_dims(normalized_data[i: i + seq_size], axis=1).tolist())
    train_y.append(normalized_data[i + 1: i + seq_size + 1].tolist())

input_dim = 1
X = tf.placeholder(tf.float32, [None, seq_size, input_dim])
Y = tf.placeholder(tf.float32, [None, seq_size])


# regression
def ass_rnn(hidden_layer_size=6):
    W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
    b = tf.Variable(tf.random_normal([1]), name='b')
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(X)[0], 1, 1])
    out = tf.matmul(outputs, W_repeated) + b
    out = tf.squeeze(out)
    return out

def train_rnn():
    out = ass_rnn()

    loss = tf.reduce_mean(tf.square(out - Y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        sess.run(tf.global_variables_initializer())

        for step in range(9000):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
            if step % 10 == 0:
                # 用测试数据评估loss
                print(step, loss_)
        print("保存模型: ", saver.save(sess, 'ass.model'))

def prediction():

    out = ass_rnn()

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        saver.restore(sess, './ass.model')

        prev_seq = train_x[-1]
        predict = []
        for i in range(12):
            next_seq = sess.run(out, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
        plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
        plt.show()


#train_rnn()
prediction()
'''