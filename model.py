import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
accuracy={}
mse={}

def train(target,df):
    data=preprocess(df,target)
    x=data.drop(columns=target,inplace=False)
    y=data[target]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=682)

    if data[target].dtype=="object":
        print("Using classification")
        model=RandomForestClassifier()
        accuracy["RFCLassifier"]=class_train_predict(model,x_train,y_train,x_test,y_test)
        model1=LogisticRegression()
        accuracy["LogisticRegression"]=class_train_predict(model1,x_train,y_train,x_test,y_test)
        model2=SVC()
        accuracy["SupportVector"]=class_train_predict(model2,x_train,y_train,x_test,y_test)
        model3=KNeighborsClassifier()
        accuracy["KNearest"]=class_train_predict(model3,x_train,y_train,x_test,y_test)
        model4=GaussianNB()
        accuracy["NaiveBayes"]=class_train_predict(model4,x_train,y_train,x_test,y_test)
    else:
        print("Using Regression")
        model=RandomForestRegressor()
        accuracy["RandomForest"]=reg_train_predict(model,x_train,y_train,x_test,y_test)
        model1=LinearRegression()
        accuracy["LinearRegression"]=reg_train_predict(model1,x_train,y_train,x_test,y_test)
        model2=Ridge()
        accuracy["RidgeRegression"]=reg_train_predict(model2,x_train,y_train,x_test,y_test)
        model3=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        accuracy["Gradient"]=reg_train_predict(model3,x_train,y_train,x_test,y_test)
        model4=Lasso()
        accuracy["LassoRegression"]=reg_train_predict(model4,x_train,y_train,x_test,y_test)
   
    #best_model=max(accuracy.values())[1]
    #for i in range(0,len(accuracy)):
    #    if accuracy[accuracy.keys()[i]][0]>=accuracy[accuracy.keys()[i+1]][0]:
    #        best_model=accuracy.values()[i][1]
    best_model=max(accuracy.values(), key=lambda x: x[0])[1]
    #if data[target].dtype=="object":
    #    acc_scr=accuracy_score(prediction,y_test)
    
    #Save Model
    joblib.dump(best_model,"./model/best_model.sav")

def reg_train_predict(model,x_train,y_train,x_test,y_test):
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc_scr=r2_score(y_pred,y_test)
    return (acc_scr,model)

def class_train_predict(model,x_train,y_train,x_test,y_test):
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc_scr=accuracy_score(y_pred,y_test)
    return (acc_scr,model)

def preprocess(data,target):
    if "Id" in data.columns:
        data.drop(columns="Id",inplace=True)
    
    if data[target].dtype!="object":
        lb=LabelEncoder()
        names=data.columns
        for i in names:
            if data[i].dtype=="object":
                data[i]=lb.fit_transform(data[i])
    return data


#data=pd.read_csv("./data/Housing.csv")
#train("price",data)