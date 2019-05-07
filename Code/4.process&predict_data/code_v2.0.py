# coding=utf-8
'''
Author:chenhao
Date: Jan 19 ,2017
Description: Data Visualization and Data Characteristics
'''
import time
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

from pylab import mpl

from scipy import stats
from scipy.stats import norm, skew

import warnings
def ignore_warn(*args , **kwargs):
    pass
warnings.warn = ignore_warn

pd.set_option('display.float_format',lambda x:'{:.3f}'.format(x))    #控制输出为精确到小数点后三位

color = sns.color_palette()
sns.set_style('darkgrid')

#导入数据
data_path = 'data/'
train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')

#将体检日期改写为天数的格式
train['体检日期'] = (pd.to_datetime(train['体检日期']) - parse('2017-09-10')).dt.days
test['体检日期'] = (pd.to_datetime(test['体检日期']) - parse('2017-09-10')).dt.days

#对性别进行转换，且其中男性比例为51%
train['性别'] = train['性别'].map({'男': 1, '女': 0, '??':0})
test['性别'] = test['性别'].map({'男': 1, '女': 0})
#print (train['性别'].describe())

#填充中位数
train_fill = train.drop(['id','性别','血糖'],axis=1)
train_fill.fillna(train_fill.median(axis=0), inplace=True)

'''
#################################################################################################
#Step1：数据集各特征缺失比例及其图像
#################################################################################################

data = pd.concat([train,test],axis=0)
null_percentage = data.isnull().sum()/len(data)
print ('The null data percentage is:',null_percentage)

#显示各特征缺失比例图像

mpl.rcParams['font.sans-serif'] = ['FangSong']

null_percentage = null_percentage.reset_index()
null_percentage.columns = ['column_name','column_value']
ind = np.arange(null_percentage.shape[0])
fig , ax = plt.subplots(figsize = (6, 8))
rects = ax.barh(ind,null_percentage.column_value.values,color='red')
ax.set_yticks(ind)
ax.set_yticklabels(null_percentage.column_name.values,rotation='horizontal')
ax.set_xlabel("各基本特征缺失数据值")
plt.show()
'''

'''
#################################################################################################
#step2：各个特征对于血糖的影响程度(使用lightgbm对特征的影响因子进行排序)
#################################################################################################

def make_feat(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    #对数据进行合并与重塑
    data = pd.concat([train, test])

    data['性别'] = data['性别'].map({'男': 1, '女': 0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-9-10')).dt.days

    data.fillna(data.median(axis=0), inplace=True)

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat

importance = train.drop(['血糖'],axis=1)
importance_name = importance.columns

train_feat, test_feat = make_feat(train, test)

predictors = [f for f in test_feat.columns if f not in ['血糖']]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('0.5mse', score, False)


print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('开始CV 5折训练...')
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
#产生相应的id数为行数5列全零数据
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'], categorical_feature=['性别'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_feat[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

print(feat_imp)

mpl.rcParams['font.sans-serif'] = ['FangSong']

feat_imp = feat_imp.reset_index()
feat_imp.columns = ['column_name','column_value']
ind = np.arange(feat_imp.shape[0])
fig , ax = plt.subplots(figsize = (6,8))
rects = ax.barh(ind,feat_imp.column_value.values,color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(feat_imp.column_name.values,rotation='horizontal')
ax.set_xlabel("各个基本特征影响权重")

plt.show()
'''

'''
#################################################################################################
#step3：各个特征对于血糖的直接影响关系(选取其中一个特征作为代表)
#################################################################################################

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
fig , ax = plt.subplots()
ax.scatter(x=train_fill['*天门冬氨酸氨基转换酶'],y=train['血糖'])
plt.ylabel('血糖')
plt.xlabel('*天门冬氨酸氨基转换酶')

soft , b = stats.boxcox(train_fill['*天门冬氨酸氨基转换酶'])
#soft += 2
fig , ax = plt.subplots()
ax.scatter(x= soft,y=train['血糖'])
plt.ylabel('血糖')
plt.xlabel('*天门冬氨酸氨基转换酶')

print(soft)
print(b)
plt.show(1)
plt.show(2)
'''

'''
#################################################################################################
#step4：各个特征对于血糖的先验高斯分布绘制图像比较
#将所有特征的基本分布情况进行展示
#################################################################################################
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

#sns.distplot(train_fill['白蛋白'],fit=norm)
#(mu,sigma) = norm.fit(train_fill['白蛋白'])
#print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
#plt.ylabel('Frequency')
#plt.title('血糖分布')
#fig1 = plt.figure()
#res1 = stats.probplot(train_fill['白蛋白'], plot=plt)

#soft , b = stats.boxcox(train_fill['*天门冬氨酸氨基转换酶'])

#train_fill['性别_log'] = np.log(train_fill['性别'])
#train_fill['*r-谷氨酰基转换酶_log'] , a = stats.boxcox(train_fill['*r-谷氨酰基转换酶'])

dist = sns.distplot(train_fill['嗜碱细胞%'],fit=norm)
(mu,sigma) = norm.fit(train_fill['嗜碱细胞%'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('嗜碱细胞%')
fig = plt.figure()
res = stats.probplot(train_fill['嗜碱细胞%'], plot=plt)

#a = train.loc[train[train['性别'] == 0 ].index]
#print(a)

plt.show()
'''

'''
#################################################################################################
#step5：尝试调用scipy中的coxbox函数对数据进行平滑处理
#################################################################################################



soft , b = stats.boxcox(train_fill['甘油三酯'])



mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

sns.distplot(soft,fit=norm)
(mu,sigma) = norm.fit(soft)
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('血糖分布')
fig = plt.figure()
res = stats.probplot(soft, plot=plt)

plt.show()
'''

'''
#################################################################################################
#step6: 探索数据关联性特征
#################################################################################################
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

corrmat = train.corr()
f,ax = plt.subplots(figsize=(15,12))
ax.set_xticklabels(corrmat,rotation='horizontal')
sns.heatmap(corrmat, vmax =0.9,square=True)
label_y = ax.get_yticklabels()
plt.setp(label_y , rotation = 360)
label_x = ax.get_xticklabels()
plt.setp(label_x , rotation = 90)

plt.show()
'''
'''
#################################################################################################
#step7: 相关变量之间的散点图
#################################################################################################
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

sns.set()
cols = ['年龄','白细胞计数','甘油三酯','红细胞平均血红蛋白浓度','尿素','尿酸']
sns.pairplot(train_fill[cols],size=2.5)
plt.show()

'''

'''
#################################################################################################
#step7: 各变量的离群值查看
#################################################################################################
#train = train.drop(train[train['*r-谷氨酰基转换酶'] > 550].index)
#train = train.drop(train[train['*丙氨酸氨基转换酶'] == 388.0].index)
a = train.loc[train[train['红细胞平均血红蛋白浓度'] > 425].index]
a = a['血糖']
print(a)

#max = a.max()
#print(max)
'''

#################################################################################################
#step8：各个特征对于血糖的直接影响关系(选取其中一个特征作为代表)
#################################################################################################

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
fig , ax = plt.subplots()
ax.scatter(x=train_fill['*天门冬氨酸氨基转换酶'],y=train['血糖'])
plt.ylabel('血糖')
plt.xlabel('*天门冬氨酸氨基转换酶')

soft , b = stats.boxcox(train_fill['*天门冬氨酸氨基转换酶'])
#soft += 2
fig , ax = plt.subplots()
ax.scatter(x= soft,y=train['血糖'])
plt.ylabel('血糖')
plt.xlabel('*天门冬氨酸氨基转换酶')

print(soft)
print(b)
plt.show(1)
plt.show(2)