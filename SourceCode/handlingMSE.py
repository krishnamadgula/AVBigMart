# handlingMSE.py
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Normalizer,PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.cross_validation import KFold,train_test_split
from sklearn.svm import LinearSVR
from xgboost import plot_importance
submission=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\AVBigMart\submission.csv')
sh_t=list(submission['Item_Outlet_Sales'].values)
for i in range(len(sh_t)):
    if sh_t[i]<=100:
        print sh_t[i]
        sh_t[i]=100
        print sh_t[i]
    if sh_t[i]>=6000:
    	sh_t[i]=5700
    	print sh_t[i]
    sh_t[i]=sh_t[i]+40	
submission.drop('Item_Outlet_Sales',axis=1)

# print 
submission['Item_Outlet_Sales']=sh_t
plt.plot(submission['Item_Outlet_Sales'])
plt.show()         
# submission.to_csv('C:\Users\Krishna\DataScienceCompetetions\AVBigMart\submission.csv')
