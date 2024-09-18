# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:53:23 2024

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

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_pred = ac.fit_predict(x)

# Data Visualization (Showing Clusters)
data_with_clusters = df.copy()
data_with_clusters['Clusters'] = y_pred
plt.scatter(data_with_clusters['Income'],data_with_clusters["TotalPurchase"], c=data_with_clusters['Clusters'],cmap='rainbow')
plt.show()


# Finding the best number of cluster with using dendogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.show()