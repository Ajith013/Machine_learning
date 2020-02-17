#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[12]:


data = pd.read_csv('iris.data')


# Now put heading according to the description mentioned in the dataset

# In[13]:


data.columns = ["sepal length", "sepal width", "petal length", "petal width", "Class"]


# In[14]:


data.head()


# Make sure that all the datatypes are correct and consistent

# In[25]:


data.info()


# Dividing the dataset in X and Y (Attributes and Classes)

# In[26]:


X = data.drop(['Class'], axis = 1)


# In[27]:


Y = data['Class']


# In[28]:


X.head()


# In[29]:


Y.head()


# Now split the data into training and test data

# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.30)


# In[44]:


classifier = GaussianNB()
classifier.fit(X_train, y_train)


# The class prior shows the probability of each class. This can be set before building the model manually. If not then it is handled by the function

# In[49]:


classifier.class_prior_


# In[45]:


y_pred = classifier.predict(X_test)


# In[46]:


cm = confusion_matrix(y_test, y_pred)


# In[56]:


print("Confusion matrix: ", cm)


# In[51]:


print("Accuracy of the model: " ,accuracy_score(y_test, y_pred))

