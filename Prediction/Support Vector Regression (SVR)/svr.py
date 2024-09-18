# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:25:57 2024

@author: Alperen
"""

# Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading Dataset
df = pd.read_csv("Fish.csv")

# Checking for missing values
df.isnull().sum()

# Choosing dependent and independent variables
y = df[["Weight"]]
x = df.drop(["Weight","Species"],axis=1)

# Handling Categorical Variables
from sklearn import preprocessing

ohe = preprocessing.OneHotEncoder()
species = ohe.fit_transform(df[["Species"]]).toarray()

species = pd.DataFrame(species,columns=df["Species"].unique())

x = pd.concat([x,species],axis=1)

# Creating the test and training set
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

# Building Model
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_sc = sc.fit_transform(x_train)

sc1 = StandardScaler()
y_sc = sc1.fit_transform(y_train)

from sklearn.svm import SVR

svr = SVR(kernel='rbf')
svr.fit(x_sc,y_sc)

# Data Visualization
plt.scatter(range(len(y_test)), sc1.fit_transform(y_test),color="#c61de3")
plt.plot(range(len(y_test)),svr.predict(sc.fit_transform(x_test)),color="#1eadc5")
# In this table, the closer the graph passes to the points (real values), the more successful the model we have built.


# We can also see the success of the model by looking at the r square score.
from sklearn.metrics import r2_score
print(r2_score(sc1.fit_transform(y_test), svr.predict(sc.fit_transform(x_test)))) # We can say that the closer the result is to 1, the more successful it is.