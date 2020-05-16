# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:35:07 2020

@author: Leeza
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('news.csv')
X = dataset.iloc[:, [1, 2]].values
y = dataset.iloc[:, 3].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 6335):
   text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
   text =text.lower()
   text =text.split()
   ps = PorterStemmer()
   text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
   text = ' '.join(text)
   corpus.append(text)
   
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 35000)
X = cv.fit_transform(corpus).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#fitting to classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

