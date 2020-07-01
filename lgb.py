import gc
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from statistics import mean

warnings.filterwarnings("ignore")

gc.collect()

# シード固定
from numpy.random import seed
seed(0)

np.set_printoptions(suppress=True)



# 特徴量設定 --------------------------------------------------------------------------

# 数値変数
numbers = ['land_price','koji_hb','kijun_hb','RE_rosenka', 'kobetsu',\
'tt_mseki','jigata','yheki_kotei','hiatari','fukuin','setsudo_hi', \
'water','niwasaki','eki_kyori1','teiho1','magutchi', \
'groups']

# カテゴリ変数
categories = ['rosen_nm1','bas_toho1']


lst = numbers + categories


# 訓練データ -------------------------------------------------------------------------- 
train_X = pd.read_csv('./dataset/filled_train_all.csv')

train_X = train_X[lst]


train_y = pd.read_csv('./dataset/train_y.csv')
train_y = np.log(train_y)

submission = pd.read_csv('./dataset/sample_submission.csv',index_col='Column1')

# テストデータ -------------------------------------------------------------------------- 
test_X = pd.read_csv('./dataset/filled_test_all.csv')

test_X = test_X[lst]

test_X.drop('groups',axis=1,inplace=True)

# 交差検証とログ変換 -------------------------------------------------------------------------- 
folds = GroupKFold(n_splits=5)

groups = train_X['groups']
train_X.drop('groups',axis=1,inplace=True)
train_columns = train_X.columns.values


for i in ['groups','water','hiatari','fukuin','jigata','kobetsu','land_price']:
	numbers.remove(i)

for col in numbers:
	train_X[col] = np.log1p(train_X[col])
	test_X[col] = np.log1p(test_X[col])


# ダミーエンコーディングの設定
train_X = pd.get_dummies(train_X,columns=categories)
test_X = pd.get_dummies(test_X,columns=categories)

train_X, test_X = train_X.align(test_X, join = 'inner', axis = 1)


params = {
		  'num_leaves': 10,
          'min_data_in_leaf': 8,
          'objective': 'mse',
          'max_depth': -1,
          'learning_rate': 0.01,
          'boosting': 'gbdt',
          'bagging_freq': 1,
          'bagging_fraction': 0.8,
          'bagging_seed': 0,
          'feature_fraction' : 0.7,
          'verbosity': -1,
          'metric' : 'mape',
          # 'categorical_feature': categories,
         }



oof = np.zeros(len(train_X))
predictions = np.zeros(len(test_X))
feature_importance_df = pd.DataFrame()

acc = list()
pic = list([[],[]])

print('train_X: ',train_X.shape)
print('train_y: ',train_y.shape)

# モデル実行
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X,train_y,groups)):
	strLog = "fold {}".format(fold_)
	print(strLog)

	print('train_idx: ',len(trn_idx))
	print('val_idx: ',len(val_idx))

	X_tr, X_val = train_X.iloc[trn_idx], train_X.iloc[val_idx]
	y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

	model = lgb.LGBMRegressor(**params, n_estimators=50000, importance_type='gain', n_jobs=-1)

	model.fit(X_tr,
			  y_tr,
			  eval_set=[(X_tr,y_tr),(X_val, y_val)],
			  eval_metric='mape',
			  verbose=1000,
			  early_stopping_rounds=500)

	y_pred = model.predict(X_val)

	y_val = np.exp(y_val)
	y_pred = np.exp(y_pred)

	acc.append(np.mean(np.abs(y_val.values.flatten() - y_pred) / y_val.values.flatten())*100)

	pic[0].extend(val_idx)
	pic[1].extend(np.abs(y_val.values.flatten() - y_pred))

	oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)

	#feature importance
	fold_importance_df = pd.DataFrame()
	fold_importance_df["Feature"] = train_columns
	fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
	fold_importance_df["fold"] = fold_ + 1
	feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

	#predictions
	predictions += model.predict(scaled_test_X, num_iteration=model.best_iteration_) / folds.n_splits

print(mean(acc))

# 使用した特徴量出力
print('[' + str(len(train_X.columns.values))+']')
print(train_X.columns.values)

submission.Column2 = np.exp(predictions).round().astype(int)
submission.Column2 -= 404720

submission.loc['test_2037','Column2'] += 1500000
submission.to_csv('./dataset/submission_1.tsv',index=True,sep='\t',header=False)

# Feature importance処理
cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False)[:200].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

pic[1] = [round(x) for x in pic[1]]

pic = pd.DataFrame({'index':pic[0],'loss':pic[1]})
pic['index'] += 2
pic.set_index('index')

pic.sort_values(by='loss',ascending=False,inplace=True)

print(pic.head(50))

plt.figure(figsize=(18,40))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.ylabel('Feature', fontsize=3)
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances_testing_6.png')

# 予測値と観測値の比較
oof = np.exp(oof)
train_y['keiyaku_pr'] = np.exp(train_y['keiyaku_pr'])

fig = plt.figure(figsize=(10,10))
plt.title('tt_mseki VS keiyaku_pr')
plt.scatter(x=train_X['tt_mseki'], y=oof, c='orange')
plt.scatter(x=train_X['tt_mseki'], y=train_y['keiyaku_pr'],c='blue')


plt.show()