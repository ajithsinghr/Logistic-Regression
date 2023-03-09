# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:25:34 2022

@author: ramav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("D:\\Assignments\\Logistic regression\\bank-full.csv",sep = ';')
df.head()
df.shape
df.isnull().sum()
list(df)
df.dtypes
df.shape

# histogram and skewness of the data
df["age"].hist()
df["age"].skew()

df["balance"].hist()
df["balance"].skew()

df["day"].hist()
df["day"].skew()

df["duration"].hist()
df["duration"].skew()

df["campaign"].hist()
df["campaign"].skew()

df["pdays"].hist()
df["pdays"].skew()

df["previous"].hist()
df["previous"].skew()
#=================================================================================
#=================================================================================
# Visualizatonn

sns.countplot(data=df,x='y')

plt.scatter(x=df["age"],y=df["balance"])

plt.scatter(x=df["day"],y=df["duration"])

plt.scatter(x=df["campaign"],y=df["pdays"])

plt.scatter(x=df["previous"],y=df["balance"])


#spliting into contineous and categorical

df_cont = df[df.columns[[0,5,9,11,12,13,14]]]


df_cat = df[df.columns[[1,2,3,4,6,7,8,10,15,16]]]


#data transformation
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for col in df_cat:
    LE=LabelEncoder()
    df_cat[col]=LE.fit_transform(df_cat[col])
df_cat.dtypes

df_new = pd.concat([df_cont,df_cat],axis=1)
df_new.head()

# Split dataset in dependent and independent variables
x=df_new.drop('y',axis=1)    
y=df_new['y']                
x.head()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_x = ss.fit_transform(x)


#==========================================================================================
#split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(ss_x,y,test_size=0.33,random_state=(42))


LogReg = LogisticRegression()

LogReg.fit(x_train,y_train)
y_pred_train = LogReg.predict(x_train)
y_pred_test = LogReg.predict(x_test)


# Generation Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_test))

from sklearn.metrics import accuracy_score
print("accuracy score for test:",accuracy_score(y_test,y_pred_test).round(2))