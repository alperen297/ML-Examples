# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:50:51 2024

@author: Alperen
"""

# Importing Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading Dataset
df = pd.read_csv("fake_bills.csv",delimiter=";")

# Checking for missing values
df.isnull().sum()

# Handling Missing Values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer = imputer.fit(df[["margin_low"]])
df["margin_low"] = imputer.transform(df[["margin_low"]])

# Handling Categorical Variables
df["is_genuine"] = df["is_genuine"].replace([True,False],[1,0])

# Data Visualization (Numbers of Real and Fake)
df["is_genuine"].value_counts().plot.barh()

# Choosing dependent and independent variables
y = df[["is_genuine"]]
x = df.drop(["is_genuine"],axis=1)

# Creating the test and training set
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

# Scaling of Data
from sklearn.preprocessing import StandardScaler  # Using scaling in the SVC algorithm gives better and more accurate results.
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Building Model
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

# Making Prediction
y_pred = svc.predict(X_test)

# Measuring the Success of the Model
from sklearn.metrics import confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))