import pickle
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
file=open('./foranalysis','rb')
list_y_pred=[]
list_y_valid=[]
Y_Pred2=[]
Y_Pred2,list_y_pred,list_y_valid=pickle.load(file)
file.close()
list_x_graph=[i for i in range(len(list_y_pred))]
plt.plot(list_x_graph,list_y_pred,'ro',list_x_graph,list_y_valid,'bo')
plt.show()
list_x_graph=[i for i in range(len(Y_Pred2))]

plt.plot(list_x_graph,Y_Pred2,'bo')
plt.show()