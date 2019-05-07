# coding=utf-8
'''
Author:chenhao
Date: Jan 25 ,2017
Description: Create the new dataset ,drop delete & log & PolynomialFeature
'''

import datetime
import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn.preprocessing import PolynomialFeatures

'''
#############################################################################
#添加Poly部分
#############################################################################

data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_for_B.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_for_B.csv', encoding='gb2312')


train['性别'] = train['性别'].map({'男': 1, '女': 0, '??':0})
test['性别'] = test['性别'].map({'男': 1, '女': 0, '??':0})

train['体检日期'] = (pd.to_datetime(train['体检日期']) - parse('2017-9-10')).dt.days
test['体检日期'] = (pd.to_datetime(test['体检日期']) - parse('2017-9-10')).dt.days

train = train.drop(['id','性别','乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)
test = test.drop(['id', '性别','乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)

# 删除缺少比较较少的参数的行
train = train.drop(train.loc[train['红细胞计数'].isnull()].index)
train = train.drop(train.loc[train['红细胞平均体积'].isnull()].index)
train = train.drop(train.loc[train['红细胞平均血红蛋白浓度'].isnull()].index)
train = train.drop(train.loc[train['白细胞计数'].isnull()].index)

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
'''

'''
#############################################################################
#计算中性粒细胞与淋巴细胞的比例
#############################################################################
Neutrophils = train['中性粒细胞%'].as_matrix()
lymphocyte = train['淋巴细胞%'].as_matrix()
#print(Neutrophils)
Ratio = Neutrophils/lymphocyte
Ratio = pd.DataFrame(Ratio)
print(Ratio)
submission = pd.DataFrame({'pred': Ratio.mean(axis=1)})
submission.to_csv(r'Ratio_train_for_B.csv', header=None,index=False, float_format='%.5f')
'''

'''
train_Poly = train.drop(['血糖'], axis=1)
test_Poly = test


array_train = np.array(train_Poly)
poly = PolynomialFeatures(2,interaction_only=True)
train_trans = poly.fit_transform(array_train)
shape = np.shape(train_trans)
print('train_poly:',shape)

array_test = np.array(test_Poly)
poly = PolynomialFeatures(2,interaction_only=True)
test_trans = poly.fit_transform(array_test)
shape = np.shape(test_trans)
print('test_poly:',shape)

df_train = pd.DataFrame(train_trans)
df_test = pd.DataFrame(test_trans)

df_train.to_csv(data_path+'train_Drop_Delete_Log_Poly_for_B.csv')
df_test.to_csv(data_path+'test_Drop_Delete_Log_Poly_for_B.csv')
'''

'''
#############################################################################
#平滑处理部分
#############################################################################

data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_for_B.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_for_B.csv', encoding='gb2312')


train['性别'] = train['性别'].map({'男': 1, '女': 0, '??':0})
test['性别'] = test['性别'].map({'男': 1, '女': 0, '??':0})

train['体检日期'] = (pd.to_datetime(train['体检日期']) - parse('2017-9-10')).dt.days
test['体检日期'] = (pd.to_datetime(test['体检日期']) - parse('2017-9-10')).dt.days

train = train.drop(['id', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体','性别'], axis=1)
test = test.drop(['id', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体','性别'], axis=1)

# 删除缺少比较较少的参数的行
train = train.drop(train.loc[train['红细胞计数'].isnull()].index)
train = train.drop(train.loc[train['红细胞平均体积'].isnull()].index)
train = train.drop(train.loc[train['红细胞平均血红蛋白浓度'].isnull()].index)
train = train.drop(train.loc[train['白细胞计数'].isnull()].index)

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

train['*r-谷氨酰基转换酶_log'] = np.log(train['*r-谷氨酰基转换酶'])
train['白细胞计数_log'] = np.log(train['白细胞计数'])
train['*丙氨酸氨基转换酶_log'] = np.log(train['*丙氨酸氨基转换酶'])
train['单核细胞%_log'] = np.log(train['单核细胞%'])
train['甘油三酯_log'] = np.log(train['甘油三酯'])
train['红细胞平均体积_log'] = np.log(train['红细胞平均体积'])
train['红细胞平均血红蛋白量_log'] = np.log(train['红细胞平均血红蛋白量'])
train['红细胞平均血红蛋白浓度_log'] = np.log(train['红细胞平均血红蛋白浓度'])
train['红细胞体积分布宽度_log'] = np.log(train['红细胞体积分布宽度'])
train['肌酐_log'] = np.log(train['肌酐'])
train['*碱性磷酸酶_log'] = np.log(train['*碱性磷酸酶'])
train['尿素_log'] = np.log(train['尿素'])
train['尿酸_log'] = np.log(train['尿酸'])
train['*球蛋白_log'] = np.log(train['*球蛋白'])
#train['嗜碱细胞%_log'] = np.log(train['嗜碱细胞%'])
#train['嗜酸细胞%_log'] = np.log(train['嗜酸细胞%'])
train['*天门冬氨酸氨基转换酶_log'] = np.log(train['*天门冬氨酸氨基转换酶'])
train['血红蛋白_log'] = np.log(train['血红蛋白'])
train['血小板比积_log'] = np.log(train['血小板比积'])
train['血小板计数_log'] = np.log(train['血小板计数'])
train['血小板体积分布宽度_log'] = np.log(train['血小板体积分布宽度'])
train['总胆固醇_log'] = np.log(train['总胆固醇'])
train['*总蛋白_log'] = np.log(train['*总蛋白'])

test['*r-谷氨酰基转换酶_log'] = np.log(test['*r-谷氨酰基转换酶'])
test['白细胞计数_log'] = np.log(test['白细胞计数'])
test['*丙氨酸氨基转换酶_log'] = np.log(test['*丙氨酸氨基转换酶'])
test['单核细胞%_log'] = np.log(test['单核细胞%'])
test['甘油三酯_log'] = np.log(test['甘油三酯'])
test['红细胞平均体积_log'] = np.log(test['红细胞平均体积'])
test['红细胞平均血红蛋白量_log'] = np.log(test['红细胞平均血红蛋白量'])
test['红细胞平均血红蛋白浓度_log'] = np.log(test['红细胞平均血红蛋白浓度'])
test['红细胞体积分布宽度_log'] = np.log(test['红细胞体积分布宽度'])
test['肌酐_log'] = np.log(test['肌酐'])
test['*碱性磷酸酶_log'] = np.log(test['*碱性磷酸酶'])
test['尿素_log'] = np.log(test['尿素'])
test['尿酸_log'] = np.log(test['尿酸'])
test['*球蛋白_log'] = np.log(test['*球蛋白'])
#test['嗜碱细胞%_log'] = np.log(test['嗜碱细胞%'])
#test['嗜酸细胞%_log'] = np.log(test['嗜酸细胞%'])
test['*天门冬氨酸氨基转换酶_log'] = np.log(test['*天门冬氨酸氨基转换酶'])
test['血红蛋白_log'] = np.log(test['血红蛋白'])
test['血小板比积_log'] = np.log(test['血小板比积'])
test['血小板计数_log'] = np.log(test['血小板计数'])
test['血小板体积分布宽度_log'] = np.log(test['血小板体积分布宽度'])
test['总胆固醇_log'] = np.log(test['总胆固醇'])
test['*总蛋白_log'] = np.log(test['*总蛋白'])

shape = np.shape(train)
print('train:',shape)

train.to_csv(data_path+'train_Drop_Delete_Log_for_B.csv')
test.to_csv(data_path+'test_Drop_Delete_Log_for_B.csv')
'''


#############################################################################
#产生Origin_Ratio_train/test_for_B.csv
#############################################################################

data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_for_B.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_for_B.csv', encoding='gb2312')


train['性别'] = train['性别'].map({'男': 1, '女': 0, '??':0})
test['性别'] = test['性别'].map({'男': 1, '女': 0, '??':0})

train['体检日期'] = (pd.to_datetime(train['体检日期']) - parse('2017-9-10')).dt.days
test['体检日期'] = (pd.to_datetime(test['体检日期']) - parse('2017-9-10')).dt.days

train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

train.fillna(train.median(axis=0), inplace=True)
test.fillna(test.median(axis=0), inplace=True)

shape = np.shape(train)
print('train:',shape)

#train.to_csv(data_path+'train_Origin_Ratio_keras_for_B.csv')
#test.to_csv(data_path+'test_Origin_Ratio_keras_for_B.csv')

#############################################################################
#计算中性粒细胞与淋巴细胞的比例
#############################################################################
Neutrophils = test['中性粒细胞%'].as_matrix()
lymphocyte = test['淋巴细胞%'].as_matrix()
#print(Neutrophils)
Ratio = Neutrophils/lymphocyte
Ratio = pd.DataFrame(Ratio)
print(Ratio)
submission = pd.DataFrame({'pred': Ratio.mean(axis=1)})
submission.to_csv(r'Ratio_test_keras_for_B.csv', header=None,index=False, float_format='%.5f')
