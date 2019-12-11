# Databricks notebook source
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
import time


df = pd.read_csv('344_blend_pl_sos_impute_sample.csv')
X = df.drop('target',axis=1)
y = df.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


scaler = StandardScaler()
imputer = Imputer()


X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)



params = {
    'n_estimators' : [1,10,50,100],
    'max_features' : ['auto',5,10,20],
    'max_depth' : [1,10,20,None],
    'class_weight' : ['balanced',{0:1,1:5},{0:1,1:10}] 
}



model = GridSearchCV(RandomForestClassifier(random_state=42,n_jobs=-1),params, n_jobs=-1)
st = time.time()
model.fit(X_train, y_train)
print(time.time()-st)