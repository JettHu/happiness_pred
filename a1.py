import os
import time
import pandas as pd
import numpy as np
#import lightgbm as lgb
import seaborn as sns

import matplotlib
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error,mean_absolute_error, f1_score
import lightgbm as lgb
import xgboost
import os
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import  KFold, StratifiedKFold,GroupKFold, RepeatedKFold
import logging

matplotlib.rcParams['font.family']='STSong'
matplotlib.rcParams['font.size']=10
pd.set_option('display.max_columns', None)

train = pd.read_csv("D:/pythonwork/datamining/data_mining/happiness_train_complete.csv", parse_dates=["survey_time"], encoding='latin-1')
test = pd.read_csv("D:/pythonwork/datamining/data_mining/happiness_test_complete.csv", parse_dates=["survey_time"], encoding='latin-1')

train_data = train[train["happiness"]!=-8].reset_index(drop=True)
print('train shape:',train_data.shape)
train_data_copy = train_data.copy()
target_col = "happiness"
target = train_data_copy[target_col]#.apply(lambda x:np.log1p(x))
del train_data_copy[target_col]

train_shape = train_data.shape[0]
data = pd.concat([train_data_copy,test],axis=0,ignore_index=True)
#f,ax=plt.subplots(1,2,figsize=(10,8))
#sns.countplot('depression',hue='happiness',data=train)
#ax[1].set_title('沮丧频繁度与幸福度')
#plt.show()

train=data
train=train.drop(["edu_other"], axis=1)



#填充数据
train["hukou_loc"]=train["hukou_loc"].fillna(1)
train["social_neighbor"]=train["social_neighbor"].fillna(8)
train["social_friend"]=train["social_friend"].fillna(8)
train["edu_status"]=train["edu_status"].fillna(5)
train["edu_yr"]=train["edu_yr"].fillna(-2)
train["work_status"]=train["work_status"].fillna(0)
train["work_yr"]=train["work_yr"].fillna(0)


train["minor_child"]=train["minor_child"].fillna(0)
train["marital_1st"]=train["marital_1st"].fillna(0)
train["s_birth"]=train["s_birth"].fillna(0)
train["marital_now"]=train["marital_now"].fillna(0)
train["s_edu"]=train["s_edu"].fillna(0)
train["work_type"]=train["work_type"].fillna(0)
train["work_manage"]=train["work_manage"].fillna(0)
train["family_income"]=train["family_income"].fillna(-2)

train["s_political"]=train["s_political"].fillna(0)
train["s_hukou"]=train["s_hukou"].fillna(0)
train["s_income"]=train["s_income"].fillna(0)
train["s_work_exper"]=train["s_work_exper"].fillna(0)
train["s_work_status"]=train["s_work_status"].fillna(0)
train["s_work_type"]=train["s_work_type"].fillna(0)

data=train

data.loc[data['health_problem']<0,'health_problem'] = 0
data.loc[data['religion']<0,'religion'] = 1
data.loc[data['religion_freq']<0,'religion_freq'] = 1
data.loc[data['edu']<0,'edu'] = 0
data.loc[data['edu_status']<0,'edu_status'] = 0
data.loc[data['income']<0,'income'] = 0
data.loc[data['s_income']<0,'s_income']= 0


data.loc[data['family_income']<=0,'family_income']=0
data.loc[data['inc_exp']<=0,'inc_exp']= 0

data.loc[data['equity']<0,'equity'] = 0
data.loc[data['social_neighbor']<0,'social_neighbor'] = 0

data.loc[data['class_10_after']<0,'class_10_after'] = 0
data.loc[data['class_10_before']<0,'class_10_before'] = 0
data.loc[data['class']<0,'class'] = 0
data.loc[data['class_14']<0,'class_14'] = 0
data.loc[data['family_m']<0,'family_m'] = 1

data.loc[data['health']<0,'health'] = 0


data.loc[data['edu_yr']<0,'edu_yr'] = 0
data['survey_time'] = pd.to_datetime(data['survey_time'])
data.loc[data['marital_1st']<0,'marital_1st']= np.nan
data.loc[data['marital_now']<0,'marital_now']= np.nan


data.loc[data['join_party']<0,'join_party']= np.nan



#配偶年龄
data['marital_sage'] = data['marital_now']-data['s_birth']
#毕业年龄
data['eduage'] = data['edu_yr'] - data['birth']
#被调查者年龄
data['survey_age'] = 2015-data['birth']
#最近结婚年龄
data['marital_age'] = data['marital_now'] - data['birth']





#收入比
data['income/s_income'] = data['income']/(data['s_income']+1)
data['income/family_income'] = data['income']/(data['family_income']+1)




#等级
data['class_10_a'] = (data['class_10_after'] - data['class_10_before'])
data['class_a'] = data['class'] - data['class_10_before']
data['class_14_a'] = data['class'] - data['class_14']
print(data)

data=data.drop(["social_neighbor "], axis=1)
data=data.drop(["work_exper"], axis=1)
data=data.drop(["inc_ability"], axis=1)



use_fea = [clo for clo in data.columns if clo!='survey_time' and data[clo].dtype!=object]
features = data[use_fea].columns
X_train = data[:train_shape][use_fea].values
y_train = target
X_test = data[train_shape:][use_fea].values

param = {
'num_leaves': 80,
'min_data_in_leaf': 40,
'objective':'regression',
'max_depth': -1,
'learning_rate': 0.01,
"min_child_samples": 30,
"boosting": "gbdt",
"feature_fraction": 0.8,
"bagging_freq": 2,
"bagging_fraction": 0.8,
"bagging_seed": 2020,
"metric": 'mse',
"lambda_l1": 0.1,
"lambda_l2": 0.2,
"verbosity": -1}
folds = KFold(n_splits=10, shuffle=True, random_state=1100)
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros(len(X_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    verbose_eval=200, early_stopping_rounds = 200)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

e=mean_squared_error(target.values, oof_lgb)
print(e)
submit = pd.read_csv("dataset/happiness_submit.csv")
submision  = pd.DataFrame({"id":submit['id'].values})
submision["happiness"]=predictions_lgb
submision.to_csv("./baseline.csv" , index=False)
submision.head(5)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def lin_train():
    lin_reg = LinearRegression()
    lin_reg.fit(X_train[:7000],y_train[:7000])

    lin_mse = mean_squared_error(y_train[7000:],lin_reg.predict(X_train[7000:]))
#     lin_rmse = np.sqrt(lin_mse)
    print(lin_mse)
    return lin_reg


def tree_train():
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X_train[:7000],y_train[:7000])

    tree_mse = mean_squared_error(y_train[7000:], tree_reg.predict(X_train[7000:]))
#     tree_rmse = np.sqrt(tree_mse)
    print(tree_mse)
    return tree_reg


def forest_train():
    #随机森林
    forest_reg = RandomForestRegressor()
    forest_reg.fit(X_train[:7000],y_train[:7000])

    forest_mse = mean_squared_error(y_train[7000:], forest_reg.predict(X_train[7000:]))
#     forest_rmse = np.sqrt(forest_mse)
    print(forest_mse)
    return forest_reg

lin_model = lin_train()
submision["happiness"]=lin_model.predict(X_test)
submision.to_csv("./lin_model.csv" , index=False)
submision.head(5)

dt_model = tree_train()
submision["happiness"]=dt_model.predict(X_test)
submision.to_csv("./dt_model.csv" , index=False)
submision.head(5)

rf_model = forest_train()
submision["happiness"]=rf_model.predict(X_test)
submision.to_csv("./rf_model.csv" , index=False)
submision.head(5)