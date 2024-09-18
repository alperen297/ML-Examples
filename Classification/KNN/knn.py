# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:26:00 2024

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

# Data Visualization (Distribution of Classes)
from matplotlib import cm
from pandas.plotting import scatter_matrix

colors = y_train.replace([0,1],["#41ccbf","#308f8e"])
colors = list(colors["is_genuine"])
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(x_train,c=colors, marker= 'x', s= 40, hist_kwds={'bins': 15}, figsize= (6,6), cmap = cmap)


# Building Model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# Measuring the Success of the Model
knn.score(x_test, y_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, knn.predict(x_test)))


# To find out which k value gives the best results
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
# It is observed in the graph that the best k value is 5. (There is no difference between 5 or greater than 5. They can also be used.)