# coding:utf-8
#Author: chenhao
#date: Jan.24.2018
#Description: Tianchi Medical solution using Features (LGB)


import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')
train_lgb = pd.read_csv(data_path + 'sub20180124_142217.csv', encoding='gb2312')
train_xgb = pd.read_csv(data_path + 'sub20180124_143226.csv', encoding='gb2312')
train_keras = pd.read_csv(data_path + 'sub20180124_153450.csv', encoding='gb2312')
test_lgb = pd.read_csv(data_path + 'lgb.csv', encoding='gb2312')
test_xgb = pd.read_csv(data_path + 'xgb.csv', encoding='gb2312')
test_keras = pd.read_csv(data_path + 'keras.csv', encoding='gb2312')

def make_feat(train, test):

    train['性别'] = train['性别'].map({'男': 1, '女': 0, '??':0})
    test['性别'] = test['性别'].map({'男': 1, '女': 0, '??': 0})
    train['体检日期'] = (pd.to_datetime(train['体检日期']) - parse('2017-9-10')).dt.days
    test['体检日期'] = (pd.to_datetime(test['体检日期']) - parse('2017-9-10')).dt.days


    # 对数据缺失值进行处理
    train_y = train['血糖']
    train = train.drop(['id', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体','血糖'], axis=1)
    test = test.drop(['id', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)

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

    train_feat = train
    test_feat = test

    return train_feat, test_feat

train_feat, test_feat = make_feat(train, test)


#将三个模型的训练集train_x进行合并
#train_feat['1'] = pd.Series(train_lgb['1'],index=train_lgb.index)
#train_feat['2'] = pd.Series(train_xgb['1'],index=train_xgb.index)
#train_feat['3'] = pd.Series(train_keras['1'],index=train_keras.index)
#test_feat['1'] = pd.Series(test_lgb['1'],index=test_lgb.index)
#test_feat['2'] = pd.Series(test_xgb['1'],index=test_xgb.index)
#test_feat['3'] = pd.Series(test_keras['1'],index=test_keras.index)

#shape = np.shape(train_feat)
#print(shape)
#print(train_feat['甘油三酯'])

array = np.array(train_feat)
poly = PolynomialFeatures(2,interaction_only=True)
train_feat_trans = poly.fit_transform(array)
shape = np.shape(train_feat_trans)
print(shape)

array1 = np.array(test_feat)
poly = PolynomialFeatures(2,interaction_only=True)
test_feat_trans = poly.fit_transform(array1)
shape1 = np.shape(test_feat_trans)
print(shape1)

#test_preds = np.zeros((train_x.shape[0],1))
print(train_feat_trans)

#predictors = [f for f in test_feat_trans.columns if f not in ['血糖']]

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
train_preds = np.zeros(train_feat_trans.shape[0])
#产生相应的id数为行数5列全零数据
test_preds = np.zeros((test_feat_trans.shape[0], 5))
kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat_trans.iloc[train_index]
    train_feat2 = train_feat_trans.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1, train_y, categorical_feature=['性别'])
    lgb_train2 = lgb.Dataset(train_feat2, train_y)
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    #feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2)
    test_preds[:, i] = gbm.predict(test_feat_trans)
print('线下得分：    {}'.format(mean_squared_error(train_y, train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False, float_format='%.4f')
