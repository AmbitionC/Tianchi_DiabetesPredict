# coding=utf-8
'''
Author:chenhao
Date: Jan 25 ,2017
Description: Feature engineer
'''

import time
import datetime
import numpy as np
import pandas as pd
from dateutil.parser import parse


data_path = 'data/'

train = pd.read_csv(data_path + 'train_Drop_Delete_Log.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'test_Drop_Delete_Log.csv', encoding='gb2312')


predictors = [f for f in train.columns if f not in ['血糖']]

print(train[predictors])

'''
def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('0.5mse', score, False)


print('开始训练...')

#原版调参数据
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

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,index=False, float_format='%.3f')
'''