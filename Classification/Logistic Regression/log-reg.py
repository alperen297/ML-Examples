# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:30:23 2024

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

# Metrics
import statsmodels.api as sm
loj = sm.Logit(y,x)
loj_model = loj.fit() 
loj_model.summary()

# When we look at the p values, we see that the p value of the diagonal variable is higher than 0.05.
# This means that this variable is meaningless for the model and should not be used in the model.

# If you remove this variable and look at the metrics again, you will see that the model has become more successful.
del x["diagonal"]

# Creating the test and training set
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#Building Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)

lr.intercept_ # alpha value
lr.coef_      # beta coefficients

# Logistic Regression Function
print(f"y = {'%0.2f' %lr.intercept_} {'%0.2f' %lr.coef_[0][0]} {list(x.columns)[0]} {'%0.2f' %lr.coef_[0][1]} {list(x.columns)[1]} {'%0.2f' %lr.coef_[0][2]} {list(x.columns)[2]} {'%0.2f' %lr.coef_[0][3]} {list(x.columns)[3]} +{'%0.2f' %lr.coef_[0][4]} {list(x.columns)[4]}")

# Making Prediction
y_pred = lr.predict(x_test)

# Measuring the Success of the Model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
