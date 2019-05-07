# coding=utf-8
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from tqdm import tqdm
import gc
import datetime
from dateutil.parser import parse

data_path = 'data/'

train = pd.read_csv(data_path + 'train_Drop_Delete_Log_Poly_Ratio_for_B.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'test_Drop_Delete_Log_Poly_Ratio_for_B.csv', encoding='gb2312')


train_df = train
test_df = test

exclude_other = ['血糖']

cat_feature_inds = []


X_train = train_df.drop(['血糖'],axis=1)
y_train = train_df['血糖']
X_test = test_df

#feature_num = 803
for i in X_train.columns:
    cat_feature_inds.append(i)

print(cat_feature_inds)

num_ensembles = 5
y_pred = 0.0

for i in tqdm(range(num_ensembles)):
    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=i)
    model.fit(X_train, y_train)
    y_pred += model.predict(X_test)


y_pred /= num_ensembles

print (y_pred)

submission = pd.DataFrame({'pred': y_pred})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,index=False, float_format='%.4f')
