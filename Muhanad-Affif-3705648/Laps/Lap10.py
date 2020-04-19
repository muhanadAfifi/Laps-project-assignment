#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


from sklearn import datasets

dataset= datasets.load_iris()
df= pd.DataFrame(dataset['data'], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

df


# In[4]:


X= df['sepal length (cm)']
Y= df['petal length (cm)']
Slic_df = pd.DataFrame({'sepal length': X, 'petal length': Y})
print(Slic_df)


# In[5]:


plt.scatter(Slic_df[['sepal length']], Slic_df[['petal length']], color = "r", marker = "x", s = 15)
plt.xlabel('Sepal length') 
plt.ylabel('Petal length')
plt.title('Scatter Plot') 
plt.show()


# In[6]:


from sklearn.linear_model import LinearRegression

#define the classifier
classifier = LinearRegression()

#train the classifier
model = classifier.fit(Slic_df[['sepal length']], Slic_df[['petal length']])


# In[7]:


y_pred = classifier.predict(Slic_df[['sepal length']])
print(y_pred)

#print coefficient (a in y=ax+b) and intercept (the constant, b in y=ax+b)
print('Coefficients: \n', classifier.coef_)
print('Intercept: \n', classifier.intercept_)


# In[8]:


plt.scatter(Slic_df[['sepal length']], Slic_df[['petal length']], color = "m", marker = "s", s = 10) 
plt.plot(Slic_df['sepal length'], y_pred, color = "g") 
plt.xlabel('Sepal length') 
plt.ylabel('Petal length')
plt.title('Regression Function')
plt.show()


# In[9]:


df2 = pd.DataFrame({'col1': [19, 20,21,22,23,24,25,26,27,28,29,30,31], 
                    'col2': [36,70,48,119,51,205,133,112,92,99,96,154,110]})

classifier = LinearRegression()

classifier.fit(df2[['col1']],df2[['col2']])

y_predict = classifier.predict(df2[['col1']])

plt.scatter(df2[['col1']], df2[['col2']], color = "m", marker = "o", s = 30) 

plt.plot(df2[['col1']], y_predict, color = "g") 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title('title')
plt.show()

print('Coefficients: \n', classifier.coef_)
print('Intercept: \n', classifier.intercept_)


# What is the predicted value of the day [18.03.2020]? 
# 64.03846154 
# 
# Is your model accurate?  YES

# In[10]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Mean squared error:")
print(mean_squared_error(df2[['col2']], y_predict))


# In[11]:


plt.scatter(y_predict, (df2[['col2']] - y_predict) ** 2, color = "blue", s = 10,) 
plt.title("Squared errors")
## plotting line for zero error 
plt.hlines(y = 0, xmin = 0, xmax = 140, linewidth = 2) 
plt.xlabel('predicted values')
plt.ylabel('squared error')
plt.show()


# In[12]:


print("Mean absolute error: ")
print(mean_absolute_error(df2['col2'], y_predict))

plt.scatter(y_predict, (df2[['col2']] - y_predict), color = "blue", s = 10,) 
plt.title("Absolute Errors")
plt.hlines(y = 0, xmin = 0, xmax = 140, linewidth = 2) 
plt.xlabel('predicted values')
plt.ylabel('absolute error')
plt.show()


# In[13]:


df['Species']= dataset['target']
df['Species']=df['Species'].apply(lambda x:dataset['target_names'][x])
df['Species']


# In[14]:


from sklearn.linear_model import LogisticRegression


X = Slic_df[['sepal length']] 
y = df['Species'] 

log_regression = LogisticRegression(solver = 'liblinear', multi_class = 'ovr')
log_regression.fit(X, y)

pred = log_regression.predict(X)

print('Score: \n', log_regression.score(X, y))
print('Coefficients: \n', log_regression.coef_)
print('Intercept: \n', log_regression.intercept_)


# In[15]:


X = df.iloc[:, 0:4]
y=df['Species']
print('X: \n', X.head(), '\n')
print('y: \n', y.head(), '\n')

from sklearn.linear_model import LogisticRegression
classifier3 = LogisticRegression( solver = 'liblinear', multi_class = 'ovr')
classifier3.fit(X, y)

y_pred = classifier3.predict(X)

print('Score: \n', classifier3.score(X, y))
print('Coefficients: \n', classifier3.coef_)
print('Intercept: \n', classifier3.intercept_)

