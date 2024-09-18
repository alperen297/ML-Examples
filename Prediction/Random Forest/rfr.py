# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:23:23 2024

@author: Alperen
"""

# Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading Dataset
df = pd.read_csv("tips.csv")

# Checking for missing values
df.isnull().sum()

# Handling Categorical Variables
df["sex"] = df["sex"].replace(["Female","Male"],[0,1])
df["smoker"] = df["smoker"].replace(["No","Yes"],[0,1])
df["time"] = df["time"].replace(["Dinner","Lunch"],[0,1])

from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()

day = ohe.fit_transform(df[["day"]]).toarray()
day = pd.DataFrame(day,columns=["Fri","Sat","Sun","Thur"])

del df["day"]

df = pd.concat([df,day],axis=1)

# Choosing dependent and independent variables
y = df[["tip"]]
x = df.drop(["tip"],axis=1)

# Creating the test and training set
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#Building Model
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 10,random_state=0)
rfr.fit(x_train,y_train)

# Metrics
import statsmodels.api as sm
model = sm.OLS(rfr.predict(x_test),y_test)
model.fit().summary()

from sklearn.metrics import r2_score
r2_score(y_test, rfr.predict(x_test))

# Data Visualization
import seaborn as sns

sns.kdeplot(y_test,linewidth=2,label="Observed Values",color="#000ff8")
sns.kdeplot(rfr.predict(x_test),linewidth=2,label="Predicted Values",color="#d520ee")
plt.xlabel("Values")
plt.title("KDE Plot for Observed vs Predicted Values",color="#000ff8")
plt.legend()
# In this table, the closer the graphs passes to each other, the more successful the model we have built.
