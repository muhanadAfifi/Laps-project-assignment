#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
df = pd.read_csv ('C:\\Users\\W\\Desktop\\python\\python -project\\Frogs_MFCCs.csv')
print(df)


# In[4]:


df.info()


# In[5]:


print(df.shape)
print(df["Family"].value_counts())
df["Family"].hist()


# In[6]:


print(df.shape)
print(df.columns)


# In[7]:


df.describe()


# In[8]:


import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[9]:



sns.boxplot(x='Family',y='MFCCs_ 3', data=df) 
plt.show()


# In[10]:


sns.violinplot(x='Family',y='MFCCs_ 3', data=df, size=20)
plt.show()


# In[11]:



df.plot(kind='scatter',x='MFCCs_ 2',y='MFCCs_ 3', c= 'Y')
plt.show()


# In[12]:


sns.set_style("whitegrid");
columns = ['MFCCs_ 1', 'MFCCs_ 2', 'MFCCs_ 3',
                      'MFCCs_ 4','MFCCs_ 5', 'MFCCs_ 6', 'MFCCs_ 7', 
                      'MFCCs_ 8', 'MFCCs_ 9']
sns.pairplot(df,vars = columns, hue = 'Family', diag_kind = 'kde', 
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, size = 2) 
plt.show()


# In[13]:


df.isnull().sum()


# In[14]:


from sklearn.preprocessing import StandardScaler , MinMaxScaler
sca = ['MFCCs_ 1','MFCCs_ 2','MFCCs_ 3','MFCCs_ 4','MFCCs_ 5']

x =  df.loc[:,sca].values
y = df.loc[: , ['Family']].values

x=StandardScaler().fit_transform(x)
print(x)


# In[15]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[16]:


finalDf = pd.concat([principalDf, df[['Family']]], axis = 1)


# In[17]:


pca.explained_variance_ratio_


# In[18]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1 ', fontsize =15)
ax.set_ylabel('Principal Component 2 ', fontsize =15)
ax.set_title('2 Component PCA', fontsize = 20)

Fimiles = ['Bufonidae','Dendrobatidae','Hylidae','Leptodactylidae']
color = ['r','g','b','y']
for fimil , color in zip(Fimiles,color):
    indicesToKeep = finalDf['Family'] == fimil
    ax.scatter(finalDf.loc[indicesToKeep,'principal component 1'], finalDf.loc[indicesToKeep,'principal component 2'], c =color , s = 10)
ax.legend(Fimiles)
ax.grid()


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
x = finalDf.iloc[:, [0,1,]].values


# In[20]:


kmeans4 = KMeans(n_clusters=4)
y_kmeans4 = kmeans4.fit_predict(x)
print(y_kmeans4)
kmeans4.cluster_centers_
plt.scatter(x[:,0],x[:,1],c=y_kmeans4,cmap='viridis')


# In[21]:


import graphviz 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, export_graphviz 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
seed = 10
# Creating a LabelEncoder and fitting it to the dataset labels. 
le = LabelEncoder()
le.fit(finalDf['Family'].values)
# Converting dataset str labels to int labels. 
y = le.transform(finalDf['Family'].values)
# Extracting the instances data. 
X = df.drop('Family', axis=1).values
# Splitting into train and test sets. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, stratify=y, random_state=seed)


# In[22]:


dd = pd.DataFrame(X)


# In[23]:


ddd = dd.drop( columns=[22, 23])


# In[24]:


X = np.asarray(ddd)
X


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, stratify=y, random_state=seed)


# In[26]:


tree = DecisionTreeClassifier(criterion='gini',
                              min_samples_leaf=5,
                              min_samples_split=5,
                              max_depth=None,
                              random_state=seed) 

tree.fit(X_train, y_train) 
y_pred = tree.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))


# In[27]:


print('Confusion Matrix is') 
print(confusion_matrix(y_test, y_pred)) 
cm=confusion_matrix(y_test, y_pred)
plt.matshow(cm) 
plt.show()


# In[28]:


print(classification_report(y_test,y_pred))

