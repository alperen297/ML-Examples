# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:50:16 2024

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

# Data Visualization 
plt.pie(df["is_genuine"].value_counts(),labels=["Real","Fake"],colors=["#485252","#3ec8c9"], autopct="%1.1f%%")
plt.title("Real and Fake Bill Distrubition")
plt.show()

# Choosing dependent and independent variables
y = df[["is_genuine"]]
x = df.drop(["is_genuine"],axis=1)

# Creating the test and training set
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

# Building Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(x_train,y_train)

# Making Prediction
y_pred = rfc.predict(x_test)

# Measuring the Success of the Model
from sklearn.metrics import confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
