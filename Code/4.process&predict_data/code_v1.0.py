# coding:utf-8

#Author: chenhao
#date: Jan.14.2018
#Description: Tianchi Medical solution data vision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew

pd.set_option('display.float_format', lambda x:'{:.3f}'.format(x))

train = pd.read_csv('data/d_train_20180102.csv',encoding='gbk')
test = pd.read_csv('data/d_test_A_20180102.csv',encoding='gbk')

print('train shape',train.shape)
print('test shape',test.shape)

train_ID = train['id']
test_ID = test['id']


print('train feature shape',train.shape)
print('test feature shape',test.shape)

print(train.head())
print(test.head())

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

fig ,ax = plt.subplots()
ax.scatter(x = train['年龄'],y=train['血糖'])
plt.ylabel('血糖')
plt.xlabel('年龄')
plt.show()

sns.distplot(train['血糖'],fit=norm)

(mu,sigma) = norm.fit(train['血糖'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('血糖分布')

fig = plt.figure()
res = stats.probplot(train['血糖'], plot=plt)
plt.show()