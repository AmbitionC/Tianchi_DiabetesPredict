# coding:utf-8
'''
#Author: chenhao
#date: Jan.26.2018
#Description: describe the datasets of model using LGB & XGB 
'''

import numpy as np
import pandas as pd

data_path = 'data/'

source_data = pd.read_csv(data_path + 'train_Drop_Delete_Log.csv', encoding='gb2312')
data = pd.read_csv(data_path + 'sub20180126_213214.csv', encoding='gb2312')

##############################################################################
#统计训练集和测试集的比例
##############################################################################
data = data.drop(data[data['1'] < 8].index)
#print(data)
shape = np.shape(data)
percentage = shape[0]/10
print('train people is ',shape[0])
print('train percentage is: ',percentage,'%')

source_data = source_data.drop(source_data[source_data['血糖'] < 5.45].index)
#print(source_data)
shape1 = np.shape(source_data)
percentage1 = shape1[0]/56.16
print('test people is ',shape1[0])
print('test percentage is: ',percentage1,'%')

