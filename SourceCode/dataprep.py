import pickle
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer,StandardScaler,LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
norm=Normalizer()
train=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\AVBigMart\Train_UWu5bXk.csv')
test=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\AVBigMart\Test_u94Q5KV.csv')
# train=train.drop('Item_Identifier',axis=1)


# ha=train['Outlet_Identifier'].values.tolist()
# train['Outlet_Identifier']=[hash(i) for i in ha]
# temp=train.Outlet_Identifier.values.tolist()
# print train.Outlet_Identifier.values
# temp=norm.fit_transform(temp)
# temp=np.array(temp)
# temp=np.reshape(temp,-1)
# train['Outlet_Identifier']=temp
std=StandardScaler()
print len(train)
train.Item_Weight[pd.isnull(train['Item_Weight'])]=np.mean(train['Item_Weight'])
# train=train[train.Item_Weight.notnull()]
# print train.head
# print len(train)
# print Krishna
train.Item_Fat_Content[train['Item_Fat_Content']=='LF']=1
train.Item_Fat_Content[train['Item_Fat_Content']=='Low Fat']=1
train.Item_Fat_Content[train['Item_Fat_Content']=='reg']=2
train.Item_Fat_Content[train['Item_Fat_Content']=='Regular']=2
train.Item_Fat_Content[train['Item_Fat_Content']=='low fat']=1
train.Item_Fat_Content[pd.isnull(train['Item_Fat_Content'])]=1
train.Item_Type[train['Item_Type']=='Baking Goods']=1
train.Item_Type[train['Item_Type']=='Breads']=2
train.Item_Type[train['Item_Type']=='Breakfast']=3
train.Item_Type[train['Item_Type']=='Canned']=4
train.Item_Type[train['Item_Type']=='Dairy']=5
train.Item_Type[train['Item_Type']=='Frozen Foods']=6
train.Item_Type[train['Item_Type']=='Fruits and Vegetables']=7
train.Item_Type[train['Item_Type']=='Hard Drinks']=8
train.Item_Type[train['Item_Type']=='Health and Hygiene']=9
train.Item_Type[train['Item_Type']=='Household']=10
train.Item_Type[train['Item_Type']=='Meat']=11
train.Item_Type[train['Item_Type']=='Others']=12
train.Item_Type[train['Item_Type']=='Seafood']=13
train.Item_Type[train['Item_Type']=='Snack Foods']=14
train.Item_Type[train['Item_Type']=='Starchy Foods']=15
train.Item_Type[train['Item_Type']=='Soft Drinks']=16
train.Item_Type[pd.isnull(train['Item_Type'])]=7
train.Outlet_Establishment_Year=train['Outlet_Establishment_Year'] - 1985
train.Outlet_Size[train['Outlet_Size']=='Small']=1
train.Outlet_Size[train['Outlet_Size']=='Medium']=2
train.Outlet_Size[train['Outlet_Size']=='High']=3
train.Outlet_Size[pd.isnull(train['Outlet_Size'])]=2
train.Outlet_Location_Type[train['Outlet_Location_Type']=='Tier 1']=1
train.Outlet_Location_Type[train['Outlet_Location_Type']=='Tier 2']=2
train.Outlet_Location_Type[train['Outlet_Location_Type']=='Tier 3']=3
train.Outlet_Location_Type[pd.isnull(train['Outlet_Location_Type'])]=3
train.Outlet_Type[train['Outlet_Type']=='Grocery Store']=1
train.Outlet_Type[train['Outlet_Type']=='Supermarket Type1']=2
train.Outlet_Type[train['Outlet_Type']=='Supermarket Type2']=3
train.Outlet_Type[train['Outlet_Type']=='Supermarket Type3']=4
train.Outlet_Type[pd.isnull(train['Outlet_Type'])]=2
print len(train)
# train.Outlet_Type.dropna()
print len(train)
list_y=train['Item_Outlet_Sales'].values
train=train.drop('Item_Outlet_Sales',axis=1)
# print train.head
#

# #
# ha=test['Outlet_Identifier'].values.tolist()
# test['Outlet_Identifier']=[hash(i) for i in ha]
# temp=test.Outlet_Identifier.values.tolist()
# print test.Outlet_Identifier.values
# temp=norm.fit_transform(temp)
# temp=np.array(temp)
# temp=np.reshape(temp,-1)
# test['Outlet_Identifier']=temp
# test=test.drop('Item_Identifier',axis=1)


test.Item_Weight[pd.isnull(test['Item_Weight'])]=np.mean(test['Item_Weight'])


test.Item_Fat_Content[test['Item_Fat_Content']=='LF']=1
test.Item_Fat_Content[test['Item_Fat_Content']=='Low Fat']=1
test.Item_Fat_Content[test['Item_Fat_Content']=='low fat']=1
test.Item_Fat_Content[test['Item_Fat_Content']=='reg']=2
test.Item_Fat_Content[test['Item_Fat_Content']=='Regular']=2
test.Item_Fat_Content[pd.isnull(test['Item_Fat_Content'])]=1
test.Item_Type[test['Item_Type']=='Baking Goods']=1
test.Item_Type[test['Item_Type']=='Breads']=2
test.Item_Type[test['Item_Type']=='Breakfast']=3
test.Item_Type[test['Item_Type']=='Canned']=4
test.Item_Type[test['Item_Type']=='Dairy']=5
test.Item_Type[test['Item_Type']=='Frozen Foods']=6
test.Item_Type[test['Item_Type']=='Fruits and Vegetables']=7
test.Item_Type[test['Item_Type']=='Hard Drinks']=8
test.Item_Type[test['Item_Type']=='Health and Hygiene']=9
test.Item_Type[test['Item_Type']=='Household']=10
test.Item_Type[test['Item_Type']=='Meat']=11
test.Item_Type[test['Item_Type']=='Others']=12
test.Item_Type[test['Item_Type']=='Seafood']=13
test.Item_Type[test['Item_Type']=='Snack Foods']=14
test.Item_Type[test['Item_Type']=='Starchy Foods']=15
test.Item_Type[test['Item_Type']=='Soft Drinks']=16
test.Item_Type[pd.isnull(test['Item_Type'])]=7
test.Outlet_Establishment_Year[test['Outlet_Establishment_Year']]=test['Outlet_Establishment_Year']-1985
test.Outlet_Size[test['Outlet_Size']=='Small']=1
test.Outlet_Size[test['Outlet_Size']=='Medium']=2
test.Outlet_Size[test['Outlet_Size']=='High']=3
test.Outlet_Size[pd.isnull(test['Outlet_Size'])]=2
test.Outlet_Location_Type[test['Outlet_Location_Type']=='Tier 1']=1
test.Outlet_Location_Type[test['Outlet_Location_Type']=='Tier 2']=2
test.Outlet_Location_Type[test['Outlet_Location_Type']=='Tier 3']=3
test.Outlet_Location_Type[pd.isnull(test['Outlet_Location_Type'])]=3
test.Outlet_Type[test['Outlet_Type']=='Grocery Store']=1
test.Outlet_Type[test['Outlet_Type']=='Supermarket Type1']=2
test.Outlet_Type[test['Outlet_Type']=='Supermarket Type2']=3
test.Outlet_Type[test['Outlet_Type']=='Supermarket Type3']=4
test.Outlet_Type[pd.isnull(test['Outlet_Type'])]=2


list_r=train.Item_Identifier.values.tolist()
list_r=set(list_r)
list_r=list(list_r)
Item_count=dict()
for i in (list_r):
    
    Item_count.update({i:train.Outlet_Identifier[train['Item_Identifier']==i].count()})
list_x=[]
for i in train.Item_Identifier.values.tolist():
    list_x.append(Item_count[i])
list_x=np.array(list_x)
list_x=np.reshape(list_x,-1)

train['Item_per_Outlet']=list_x
train['Outlet_Years_of_Existence']=abs(train['Outlet_Establishment_Year']+1985-2016)
train=train.drop('Outlet_Establishment_Year',axis=1)
# train['Item_Visibility']=train['Item_Visibility']
# train['Item_MRP_per_Wt']=train['Item_MRP']/train['Item_Weight'] #apparently important
# train['Item_wt_times_visibility']=train['Item_Weight']*train['Item_Visibility']
# train['Item_visibility_times_MRP']=train['Item_Visibility']*train['Item_MRP'] #hmm even this one 
# train['Item_wt_times_MRP']=train['Item_MRP']*train['Item_Weight']
train['Outlet_loc_type']=train['Outlet_Location_Type']-train['Outlet_Type']
train['Outletyears_type']=train['Outlet_Years_of_Existence']*train['Outlet_Location_Type']
train['Item_type_fat']=train['Item_Type']+train['Item_Fat_Content']
# train['Outlet_size_type_loc']=train['Outlet_Location_Type']+train['Outlet_Type']+train['Outlet_Size']
# train['Item_visibility_times_Outlet_size']=train['Item_Visibility']*train['Outlet_Size']




outlets=dict()
list_outlets=train.Outlet_Identifier.values.tolist()
list_outlets=set(list_outlets)
list_outlets=list(list_outlets)
for i in list_outlets:
    tempo=str(i).split('T')
    
    outlets.update({i:tempo[1]})
listm=[]    
for i in train.Outlet_Identifier.values.tolist():
    listm.append(int(outlets[i]))

listm=np.array(listm)
listm=np.reshape(listm,-1)
train['Outlet_Identifier']=listm
# train=train.drop('Outlet_Identifier',axis=1)
le=LabelEncoder()


train['Item_Identifier']=le.fit_transform(train.Item_Identifier.values.tolist())

# train=train.drop('Item_Identifier',axis=1)

list_r=test.Item_Identifier.values.tolist()
list_r=set(list_r)
list_r=list(list_r)
Item_count=dict()
for i in (list_r):
    # list_temp=np.append(0)
    Item_count.update({i:test.Outlet_Identifier[test['Item_Identifier']==i].count()})
list_x=[]
for i in test.Item_Identifier.values.tolist():
    list_x.append(Item_count[i])

list_x=np.array(list_x)
list_x=np.reshape(list_x,-1)
test['Item_per_Outlet']=list_x
print test.Item_per_Outlet.values
test['Outlet_Years_of_Existence']=abs(test['Outlet_Establishment_Year']+1985-2016)
test=test.drop('Outlet_Establishment_Year',axis=1)
# test['Item_Visibility']=test['Item_Visibility']
# test['Item_MRP_per_Wt']=test['Item_MRP']/test['Item_Weight']
test['Item_type_fat']=test['Item_Type']+test['Item_Fat_Content']
# test['Item_wt_times_visibility']=test['Item_Weight']*test['Item_Visibility']
# test['Item_visibility_times_MRP']=test['Item_Visibility']*test['Item_MRP']
# test['Item_wt_times_MRP']=test['Item_MRP']*test['Item_Weight']
test['Outlet_loc_type']=test['Outlet_Location_Type']-test['Outlet_Type']
# test['Outlet_size_type_loc']=test['Outlet_Location_Type']+test['Outlet_Type']+test['Outlet_Size']

# test['Item_visibility_times_Outlet_size']=test['Item_Visibility']*test['Outlet_Size']
test['Outletyears_type']=test['Outlet_Years_of_Existence']*test['Outlet_Location_Type']
# test['Item_MRP_and_visibility']=test['Item_MRP']/test['Item_Visibility']
outlets=dict()
list_outlets=test.Outlet_Identifier.values.tolist()
list_outlets=set(list_outlets)
list_outlets=list(list_outlets)
for i in list_outlets:
    tempo=str(i).split('T')
    
    outlets.update({i:tempo[1]})
listm=[]    
for i in test.Outlet_Identifier.values.tolist():
    listm.append(int(outlets[i]))
# listm=norm.fit_transform(listm)
listm=np.array(listm)
listm=np.reshape(listm,-1)
test['Outlet_Identifier']=(listm)
# print 'HEY this is outlet id'
# print test.Outlet_Identifier.values
    
# test=test.drop('Outlet_Identifier',axis=1)

test['Item_Identifier']=le.fit_transform(test.Item_Identifier.values.tolist())
# test=test.drop('Item_Identifier',axis=1)
list_x_test=test
list_x_train=train
# print train.columns
numeric_features_index=train.dtypes[train.dtypes!='object'].index
object_features_index=train.dtypes[train.dtypes=='object'].index
# print object_features_index
# print numeric_features_index
corrcoeff =[]
for i in numeric_features_index:
    print i
    print (np.corrcoef(train[i],list_y))
# print 'Item_vis_times_outlet'    
# tvto=list(train['Item_visibility_times_Outlet_size'].values)
# print (np.corrcoef(train['Item_visibility_times_Outlet_size'],list_y))
# print corrcoeff
file=open('C:\Users\Krishna\DataScienceCompetetions\AVBigMart\\trainFile','wb')
X_Train,X_Test,Y_Train,Y_Test=train_test_split(list_x_train,list_y,test_size=0.05)
X_valid=X_Test
Y_valid=Y_Test
pickle.dump([X_Train,X_valid,Y_Train,Y_valid,list_x_test],file)
file.close()


