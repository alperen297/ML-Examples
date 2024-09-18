# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:56:35 2024

@author: Alperen
"""

# Importing Modules
import numpy as np
import pandas as pd

# Reading Dataset
commends = pd.read_csv('Restaurant_Reviews.csv')

# Downloading Stopwords
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocessing
import re

procom = []
for i in range(1000):
    commend = re.sub('[^a-zA-Z]',' ',commends['Review'][i])
    commend = commend.lower()
    commend = commend.split()
    commend = [ps.stem(word) for word in commend if not word in set(stopwords.words('english'))]
    commend = ' '.join(commend)
    procom.append(commend)
    
#Feautre Extraction
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)

x = cv.fit_transform(procom).toarray() # independent variable
y = commends.iloc[:,1].values # dependent variable

# Creating the test and training set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Building Model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Measuring the Success of the Model
knn.score(x_test, y_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, knn.predict(x_test)))


