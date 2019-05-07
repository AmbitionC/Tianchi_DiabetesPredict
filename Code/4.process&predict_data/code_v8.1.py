# coding=utf-8
""" 
Author:chenhao
Date: Jan 27 ,2018
Description: Generate the new data set, change the BS into binary 'label'
"""
import datetime
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np


def mergeToOne(X, X2):
    X3 = []
    for i in range(X.shape[0]):
        tmp = np.array([list(X.iloc[i]), list(X2[i])])
        #按列顺序把数组堆叠起来
        X3.append(list(np.hstack(tmp)))
    X3 = np.array(X3)
    return X3

data_path = 'data/'

train = pd.read_csv(data_path + 'train_Drop_Delete_Log_Ratio_Label.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'test_Drop_Delete_Log_Ratio.csv', encoding='gb2312')

# 打乱数据
train = train.sample(len(train))
y = train.label
X = train.drop(['label'], axis=1)
Y = test

# 划分训练集测试集
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)  ##test_size测试集合所占比例

##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
#X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=0)



clf = XGBClassifier(
    learning_rate=0.2,  # 默认0.3
    n_estimators=10,  # 树的个数
    max_depth=8,
    min_child_weight=10,
    gamma=0.5,
    subsample=0.75,
    colsample_bytree=0.75,
    objective='binary:logistic',  # 逻辑回归损失函数
    nthread=8,  # cpu线程数
    scale_pos_weight=1,
    reg_alpha=1e-05,
    reg_lambda=10,
    seed=1024)  # 随机种子

clf.fit(X, y)
new_feature_X = clf.apply(X)
new_feature_Y = clf.apply(Y)
print(new_feature_X)
print(new_feature_Y)
shape_x = np.shape(new_feature_X)
shape_y = np.shape(new_feature_Y)
print('shape of train: ', shape_x)
print('shape of test: ', shape_y)

new_feature_X = pd.DataFrame(new_feature_X)
new_feature_Y = pd.DataFrame(new_feature_Y)
new_feature_X.to_csv(data_path+'train_Drop_Delete_Log_Ratio_Label_GBDT1.csv')
new_feature_Y.to_csv(data_path+'test_Drop_Delete_Log_Ratio_Label_GBDT1.csv')

#submission = pd.DataFrame({'pred': new_feature_X.mean(axis=1)})
#submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False, float_format='%.3f')
