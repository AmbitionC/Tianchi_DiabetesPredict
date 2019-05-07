# coding=utf-8
""" 
Author:chenhao
Date: Jan 27 ,2018
Description: Generate the new features using the GBDT trees
"""
import datetime
import numpy as np
import pandas as pd

data_path = 'data/'

train = pd.read_csv(data_path + 'train_Drop_Delete_Log_Ratio.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'test_Drop_Delete_Log_Ratio.csv', encoding='gb2312')

#选择血糖大于等于7的，将其添加一个label
train['label'] = pd.Series(len(train['Ratio']), index=train.index)
train['label'] = 0
train.loc[train['血糖'] >= 8, 'label'] = 1
train = train.drop(['血糖'],axis=1)

train.to_csv(data_path+'train_Drop_Delete_Log_Ratio_Label.csv')