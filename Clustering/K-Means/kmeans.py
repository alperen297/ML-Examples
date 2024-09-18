# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:31:01 2024

@author: Alperen
"""

# Importing Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading Dataset
df = pd.read_csv("marketing_campaign.csv",delimiter=";")

# Checking for missing values
df.isnull().sum()

# 
ttl = []
for i in range(len(df["ID"])):
    numPurchases = df["NumDealsPurchases"][i] + df["NumWebPurchases"][i] + df["NumCatalogPurchases"][i] + df["NumStorePurchases"][i]
    ttl.append(numPurchases)
    
df["TotalPurchase"] = ttl
    
# Removing Missing Values
df = df.dropna()

# Building Model
x = df[["Income","TotalPurchase"]]

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
identified_clusters

# Data Visualization (Showing Clusters)
data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['Income'],data_with_clusters["TotalPurchase"], c=data_with_clusters['Clusters'],cmap='rainbow')
plt.show()

# Finding the best k value
wcss=[]
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# The break point helps in determining the best k value. (Here we can say k = 3)