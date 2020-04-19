#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[2]:


x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')


# In[3]:


from sklearn import datasets
iris = datasets.load_iris()
df = pd.read_csv(iris.filename)


# In[4]:


pd.read_csv(filepath_or_buffer='C:\\Users\\W\\Anaconda3\\lib\\site-packages\\sklearn\\datasets\\data\\iris.csv').head(5)


# In[5]:


pd.read_csv(iris.filename, delimiter='.').head(5)


# In[6]:


pd.read_csv(iris.filename, header=None).head(5)


# In[7]:


arr = np.array(['this', 'is', 'names', 'parameters', 'example'])
pd.read_csv(iris.filename, names=arr).head(5)


# In[8]:


iris_with_headers = pd.read_csv(iris.filename,header=0, 
                                names=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','variety'])
iris_with_headers


# In[9]:


pd.read_csv(iris.filename).info()


# in Your observation, Is there any missing values? 
# 
# No

# In[ ]:





# In[10]:


print(df.shape)


# In[11]:


print(df["150"].count())
print(df["4"].count())
print(df["setosa"].count())
print(df["versicolor"].count())
print(df["virginica"].count())


# In[12]:


df.describe()


# In[13]:


for ojha, feature in enumerate(list(iris_with_headers.columns)[:-1]):
    fg = sns.FacetGrid(iris_with_headers, hue='variety', height=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()


# In[ ]:





# In[14]:


sns.boxplot(x='variety', y='sepal length (cm)', data=iris_with_headers)
plt.show()
sns.boxplot(x='variety', y='sepal width (cm)', data=iris_with_headers)
plt.show()
sns.boxplot(x='variety', y='petal length (cm)', data=iris_with_headers)
plt.show()
sns.boxplot(x='variety', y='petal width (cm)', data=iris_with_headers)
plt.show()


# the top of the box is 75% which is Q3.
# 
# the bottom of the box is 25% which is Q1.
# 
# the line in the middle of the box is the median.
# if the median in the middle of the box means the median and the mean is the same.
# if the median under the middle of the box means the median less than the mean.
# if the median above the middle of the box means the median grater than the mean.
# 
# any point out of the shape is out layer or missing value.

# In[15]:


sns.violinplot(x='variety', y='sepal length (cm)', data=iris_with_headers)
plt.show()


# In[16]:


iris_with_headers.plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)',c='g')
plt.show()


# In[17]:


sns.set_style('whitegrid')
sns.FacetGrid(iris_with_headers, hue='variety', height=10)   .map(plt.scatter, 'sepal length (cm)', 'sepal width (cm)')   .add_legend()
plt.show()


# In[18]:


sns.set_style('whitegrid')
sns.pairplot(iris_with_headers, vars=iris.feature_names, hue='variety',
            diag_kind='kde', plot_kws={'alpha':0.6, 'edgecolor':'k'}, size=2)
plt.show()

