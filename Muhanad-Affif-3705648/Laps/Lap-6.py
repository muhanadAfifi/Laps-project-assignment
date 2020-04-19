#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import datasets
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[ ]:


dataset=datasets.load_iris()

data=pd.DataFrame(dataset['data'],columns=['Petal Length', 'Petal Width','Sepal Length','Sepal Width'])
data['Species']=dataset['target']
data['Species']=data['Species'].apply(lambda x: dataset['target_name'][x])


# In[8]:


data.head(10)


# In[9]:


data.describe()


# In[10]:


data.isnull().sum()


# In[11]:


modData = data.append({'Petal length' : np.nan , 'Petal Width' : 3.6, 'Sepal Length': 0,
                       'Sepal Width': 0.2, 'Species': 'setosa' } , ignore_index=True)
modData.describe()


# Can you determine in which column is it?
# Petal length
# What does np.nan mean?
# missing data 

# In[ ]:


modData = data.append({'Petal length' : np.nan , 'Petal Width' : 3.6, 'Sepal Length': 0,
                       'Sepal Width': 0.2, 'Species': 'setosa' } , ignore_index=True)
modData.describe()


# In[12]:


print('Columns with missing values')
print(modData.isnull().sum())
print('\n Columns with zero values')
print((modData[['Petal length','Petal Width','Sepal Length','Sepal Width','Species']]==0).sum())


# In[13]:


modData[['Petal length',
         'Petal Width',
         'Sepal Length',
         'Sepal Width','Species']] = modData[['Petal length',
                                              'Petal Width',
                                              'Sepal Length',
                                              'Sepal Width',
                                              'Species']].replace(0, np.NaN)
print('Columns with missing values')
print(modData.isnull().sum())


# In[14]:


modData.fillna(modData.mean(), inplace=True)
print(modData.isnull().sum())


# In[15]:


modData.fillna(modData.median(), inplace=True)
print(modData.isnull().sum())


# In[16]:


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
PCA_df = pd.read_csv(url,names=['Petal length','Petal Width','Sepal Length','Sepal Width','target'])


# In[17]:


from sklearn.preprocessing import StandardScaler , MinMaxScaler
features = ['Petal length','Petal Width','Sepal Length','Sepal Width']
# Separating out the features
x=  PCA_df.loc[:,features].values
# Separating out the target
y = PCA_df.loc[: , ['target']].values
# Standardizing the features
x=StandardScaler().fit_transform(x)
print(x)


# In[18]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[21]:


finalDf = pd.concat([principalDf, PCA_df[['target']]], axis = 1)


# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#sns.set(color_codes=True)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[23]:


print(pca.explained_variance_ratio_)

