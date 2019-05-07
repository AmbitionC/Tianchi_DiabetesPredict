# coding:utf-8
#Author: chenhao
#date: Jan.22.2018
#Description: Tianchi Medical solution using stacking (LGB)
import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

data_path = 'data/'

train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')


def make_feat(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    #对数据进行合并与重塑
    data = pd.concat([train, test])

    data['性别'] = data['性别'].map({'男': 1, '女': 0, '??':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-9-10')).dt.days

    #data.fillna(data.median(axis=0), inplace=True)

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    # 对数据缺失值进行处理
    train_feat = train_feat.drop(['id', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)
    test_feat = test_feat.drop(['id', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)

    # 删除缺少比较较少的参数的行
    train_feat = train_feat.drop(train_feat.loc[train_feat['红细胞计数'].isnull()].index)
    train_feat = train_feat.drop(train_feat.loc[train_feat['红细胞平均体积'].isnull()].index)
    train_feat = train_feat.drop(train_feat.loc[train_feat['红细胞平均血红蛋白浓度'].isnull()].index)
    train_feat = train_feat.drop(train_feat.loc[train_feat['白细胞计数'].isnull()].index)

    # 对缺少一部分的数据进行填充
    train_feat.fillna(train_feat.median(axis=0), inplace=True)
    test_feat.fillna(test_feat.median(axis=0), inplace=True)

    # 删除离群值
    train_feat = train_feat.drop(train_feat[train_feat['*r-谷氨酰基转换酶'] > 600].index)
    train_feat = train_feat.drop(train_feat[train_feat['白细胞计数'] > 20.06].index)
    train_feat = train_feat.drop(train_feat[train_feat['*丙氨酸氨基转换酶'] == 498.89].index)
    train_feat = train_feat.drop(train_feat[train_feat['单核细胞%'] > 20].index)
    train_feat = train_feat.drop(train_feat[train_feat['*碱性磷酸酶'] > 340].index)  # 有待调整
    train_feat = train_feat.drop(train_feat[train_feat['*球蛋白'] > 60].index)
    train_feat = train_feat.drop(train_feat[train_feat['嗜酸细胞%'] > 20].index)
    train_feat = train_feat.drop(train_feat[train_feat['*天门冬氨酸氨基转换酶'] > 300].index)
    train_feat = train_feat.drop(train_feat[train_feat['血小板计数'] > 700].index)
    train_feat = train_feat.drop(train_feat[train_feat['*总蛋白'] > 100].index)

    return train_feat, test_feat







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
test_preds = np.zeros((train_feat.shape[0], 5))
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
    test_preds[:, i] = gbm.predict(train_feat[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.4f')
