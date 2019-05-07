# coding=utf-8
#catboost源代码测试

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from catboost import CatBoostRegressor

from tqdm import tqdm

import gc

import datetime

from dateutil.parser import parse

data_path = 'data/'


train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')

test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')


def make_feat(train, test):
    train_id = train.id.values.copy()

    test_id = test.id.values.copy()

    data = pd.concat([train, test])

    data['性别'] = data['性别'].map({'男': 1, '女': 0})

    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days

    #    data.fillna(data.median(axis=0),inplace=True)

    train_feat = data[data.id.isin(train_id)]

    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat


train_df, test_df = make_feat(train, test)

print('Remove missing data fields ...')

missing_perc_thresh = 0.98

exclude_missing = []

num_rows = train_df.shape[0]

for c in train_df.columns:

    num_missing = train_df[c].isnull().sum()

    if num_missing == 0:
        continue

    missing_frac = num_missing / float(num_rows)

    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)

print("We exclude: %s" % len(exclude_missing))

del num_rows, missing_perc_thresh

gc.collect();

print("Remove features with one unique value !!")

exclude_unique = []

for c in train_df.columns:

    num_uniques = len(train_df[c].unique())

    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1

    if num_uniques == 1:
        exclude_unique.append(c)

print("We exclude: %s" % len(exclude_unique))

print("Define training features !!")

exclude_other = ['id', '血糖', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']

train_features = []

for c in train_df.columns:

    if c not in exclude_missing \
 \
            and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)

print("We use these for training: %s" % len(train_features))

print("Define categorial features !!")

cat_feature_inds = []

cat_unique_thresh = 10

for i, c in enumerate(train_features):

    num_uniques = len(train_df[c].unique())

    if num_uniques < cat_unique_thresh:
        cat_feature_inds.append(i)

print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

print("Replacing NaN values by -999 !!")

train_df.fillna(-999, inplace=True)

test_df.fillna(-999, inplace=True)

print("Training time !!")

X_train = train_df[train_features]

y_train = train_df['血糖']

print(X_train.shape, y_train.shape)

X_test = test_df[train_features]

print(X_test.shape)

num_ensembles = 5

y_pred = 0.0

for i in tqdm(range(num_ensembles)):
    model = CatBoostRegressor(

        iterations=1000, learning_rate=0.03,

        depth=6, l2_leaf_reg=3,

        loss_function='RMSE',

        eval_metric='RMSE',

        random_seed=i)

    model.fit(

        X_train, y_train,

        cat_features=cat_feature_inds)

    y_pred += model.predict(X_test)

y_pred /= num_ensembles

submission = pd.DataFrame({'pred': y_pred})

submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,

                  index=False, float_format='%.4f')

