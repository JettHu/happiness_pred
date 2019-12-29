import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer,LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils import check_array
from scipy import sparse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib

HAPPY_PATH = "data_mining/happiness"
def load_HAPPY_data(happy_path=HAPPY_PATH):
    datatrain = pd.read_csv(happy_path+'/happiness_train_abbr.csv')
    datatest = pd.read_csv(happy_path+'/happiness_test_abbr.csv')
    return datatrain,datatest
datatrain , datatest = load_HAPPY_data()

pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)
#print(datatrain.head())
#print(datatrain.info())
#print(datatrain.describe())
#datatrain.hist(bins = 50, figsize=(20,20))
#plt.show()
datatrain["family_per_income"]=datatrain["family_income"]/datatrain["family_m"]
datatrain["age"]=datatrain["birth"]-datatrain["birth"].min()
#datatrain["city_income"]=datatrain["income"]/datatrain["city"]
#datatrain["edu_income"]=datatrain["income"]/datatrain["edu"]
#datatrain["family_per_area"]=datatrain["floor_area"]/datatrain["family_m"]
datatrain_labels = datatrain["happiness"].copy()
datatrain_num = datatrain.drop("survey_time",axis=1)
datatrain_num = datatrain_num.drop("id",axis=1)
datatrain_num = datatrain_num.drop("birth",axis=1)
datatrain_num = datatrain_num.drop("happiness",axis=1)
imputer =Imputer(strategy="median")
imputer.fit(datatrain_num)
X = imputer.transform(datatrain_num)
datatrain_tr = pd.DataFrame(X,columns=datatrain_num.columns)

#corr_matrix = datatrain_tr.corr()
#print(datatrain_tr.head())
#print(corr_matrix["happiness"].sort_values(ascending=False))
#print(datatrain_tr.info())
def lin_train():
    lin_reg = LinearRegression()
    lin_reg.fit(datatrain_tr,datatrain_labels)

    lin_mse = mean_squared_error(datatrain_labels,lin_reg.predict(datatrain_tr))
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    return lin_reg

def tree_train():
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(datatrain_tr, datatrain_labels)

    tree_mse = mean_squared_error(datatrain_labels, tree_reg.predict(datatrain_tr))
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)
    return tree_reg

def forest_train():
    #随机森林


    forest_reg = RandomForestRegressor()
    forest_reg.fit(datatrain_tr,datatrain_labels)

    forest_mse = mean_squared_error(datatrain_labels, forest_reg.predict(datatrain_tr))
    forest_rmse = np.sqrt(forest_mse)
    print(forest_rmse)
    return forest_reg

lin_train()
tree_train()
forest_train()

#print("Predictions:\t",lin_reg.predict(datatrain_tr.iloc[:5]))
#print("Labels:\t\t",list(datatrain_labels.iloc[:5]))