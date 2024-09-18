# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:01:50 2024

@author: Alperen
"""

# Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading Dataset
df = pd.read_csv("Salary_dataset.csv")

# Checking for missing values
df.isnull().sum()

# Choosing dependent and independent variables
experience = df[["YearsExperience"]] 
salary = df[["Salary"]]

# Creating the test and training set
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(experience,salary,test_size=0.33, random_state=0)


# Building Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

# Making predictions with test data
y_pred = lr.predict(x_test)  # Compare with y_test. If the values are close to each other, we can say that the predictions are successful.


# Data Visualization
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train,color="#0d132e")
plt.plot(x_test,lr.predict(x_test),color="#2e93ba")

plt.xlabel("Experiences")
plt.ylabel("Salaries")
plt.title("Salaries According to Experiences")