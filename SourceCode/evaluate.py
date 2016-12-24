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
def runner():
    file=open('C:\Users\Krishna\DataScienceCompetetions\AVBigMart\\trainFile','rb')
    X_Train,X_Valid,Y_Train,Y_valid,X_Test=pickle.load(file)

    # regr=AdaBoostRegressor(DecisionTreeRegressor(),n_estimators=100)
    # regr=LinearRegression(n_jobs=-1)
    # regr=LogisticRegression()
    # polynomials=PolynomialFeatures(degree=3,interaction_only=True)
    # polynomials.fit_transform(X_Train)
    # polynomials.fit_transform(X_Test)
    # polynomials.fit_transform(X_Valid)
    # X_Test=list(X_Test)
    regr0=RandomForestRegressor(n_estimators=100,max_depth=3,n_jobs=-1)
    regr2=GradientBoostingRegressor(n_estimators=100,max_depth=2,)
    print type(X_Train)
    regr4=XGBRegressor(n_estimators=100)
    xg_train=xgb.DMatrix(X_Train.values,label=Y_Train)
    # print xg_train #,label=Y_Train)
    xg_test=xgb.DMatrix(X_Test.values)#,label=Y_Test)
    xg_valid=xgb.DMatrix(X_Valid.values)#,label=Y_valid)
    # regr4.fit(X_Train,Y_Train)
    # Y_Pred4=regr4.predict(X_Test)
    # Y_valid_pred4=regr4.predict(X_Valid)
    # regr2=BaggingRegressor(base_estimator=regr1,n_jobs=-1,n_estimators=10)
    # regr2=BaggingRegressor(base_estimator=regr4,n_jobs=-1,n_estimators=15)
    regr5=BaggingRegressor(base_estimator=regr4,n_jobs=-1,n_estimators=10)
    # regr1=ExtraTreesRegressor(n_jobs=-1,n_estimators=100)

    list_xtrain=np.array(X_Train)
    list_xtest=np.array(X_Test)
    xgb_reg=xgb.XGBRegressor(n_estimators=100,max_depth=2)
    cv=KFold(len(X_Train),n_folds=3,shuffle=True)
    total=0     
    for train,test in cv:
        train_x=list_xtrain[train]
        train_y=Y_Train[train]
        # print train_x
        test_x=list_xtrain[test]
        print test_x
        test_y=Y_Train[test]
        # print (test_y)
        regr2.fit(train_x,train_y)
        xgb_reg.fit(train_x,train_y)
        predictions =( regr2.predict(test_x)+xgb_reg.predict(test_x))/2
        rmse=np.sqrt(mean_squared_error(predictions,test_y))
        total+=rmse
    RMSE = total / 3

    print "the rmse both ",RMSE

    # parameters={'n_estimators':[50,150,350]}
    # n_est_list=[]
    # j=25
    # while(j<250):
    #     n_est_list.append(j)
    #     j=j+25
    # models= GridSearchCV(regr1,
    #                {'max_depth': [2,4,6,10,12],
    #                 'n_estimators': n_est_list}, verbose=1,n_jobs=-1)
    # models.fit(X_Train,Y_Train)
    
    # print 'hi',models.best_params_,models.best_score_
    
    
    Y_Pred2=regr2.predict(X_Test.values)
    Y_predx=xgb_reg.predict(X_Test.values)
    print np.corrcoef(Y_Pred2,Y_predx)

    test=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\AVBigMart\Test_u94Q5KV.csv')
    # print test['Item_Identifier']
    
    # print regr.feature_importances_
    submission=pd.DataFrame()
    print submission
    submission['Item_Identifier']=0
    ItId=list(test.Item_Identifier.values)
    submission['Item_Identifier']=ItId
    print submission['Item_Identifier']
    # print submission.describe
    # print submission.values
    OtId=list(test.Outlet_Identifier.values)
    submission['Outlet_Identifier']=OtId
    submission['Item_Outlet_Sales']=(Y_predx+Y_Pred2)/2

    # submission['Purchase']=((Y_Pred1+Y_Pred2)/2)

    submission.to_csv('C:\Users\Krishna\DataScienceCompetetions\AVBigMart\submission.csv')
    plot_importance(xgb_reg)
    plt.show()
    
if __name__=='__main__':
    runner()



#   #   [ 0.05108579  0.02539443  0.07595145  0.25003104  0.18979203  0.18343231
#   # 0.04671949  0.05766914  0.11992432] gbr

#   #   [ 0.04316531  0.01727148  0.07748198  0.08758843  0.25049066  0.17606424
#   # 0.02391524  0.02700119  0.29702147] rfr