#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


from sklearn import datasets
iris = datasets.load_iris()
headersName = ['sepal length', 'sepal width', 'petal length', 'petal width']
df = pd.read_csv(iris.filename)


# In[4]:


x = df.iloc[:, [0,1,2,3]].values


# Kmeans algorithm is an interative algorithm
# and the minimum parameter is zero and it has a default value

# In[5]:


kmeans5 = KMeans(n_clusters = 5)
y_kmeans5 = kmeans5.fit_predict(x)
kmeans5.cluster_centers_


# In[6]:


kmeans2 = KMeans(n_clusters = 2)
y_kmeans2 = kmeans2.fit_predict(x)
kmeans2.cluster_centers_


# In[7]:


Error = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of cluster')
plt.ylabel('Error')
plt.show()


# From the plot above, can you determine the optimum number of clusters? 
# Yes
# Why?
# because the no of cluster 3 is between the two change points (2 and 4)

# In[8]:


plt.scatter(x[:,0], x[:,1], c = y_kmeans5, cmap= 'rainbow')


# In[9]:


kmeans = KMeans(n_clusters = 3)
y_kmeans = kmeans.fit_predict(x)
plt.scatter(x[:,0], x[:,1], c = y_kmeans, cmap= 'rainbow')

