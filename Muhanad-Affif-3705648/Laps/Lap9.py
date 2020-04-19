#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install graphviz


# In[2]:


import pandas as pd
import graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed = 10


# In[3]:


from sklearn import datasets
iris= datasets.load_iris()
df=pd.read_csv(iris.filename, delimiter=',', header= 0, 
               names= ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'Variety'])

df.head()


# In[4]:


le = LabelEncoder()
le.fit(df['Variety'].values)

y = le.transform(df['Variety'].values)

X = df.drop('Variety', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, stratify=y, random_state=seed)


# In[5]:


tree = DecisionTreeClassifier(criterion='gini',
                              min_samples_leaf=5,
                              min_samples_split=5,
                              max_depth=None,
                              random_state=seed)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))


# In[6]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

print('Confusion Matrix is')
print(confusion_matrix(y_test, y_pred))
cm=confusion_matrix(y_test, y_pred)
plt.matshow(cm)
plt.show()


# In[7]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, labels=df['Variety'].unique()))


# In[8]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


def plot_tree(tree, dataframe, label_col, label_encoder, plot_title):
    label_names = ['setosa','virginica','Versicolour']
    
    graph_data = export_graphviz(tree,
                                 feature_names=dataframe.drop(label_col, axis=1).columns,
                                 class_names=label_names,
                                 filled=True,
                                 rounded=True,
                                 out_file=None)
    
    graph = graphviz.Source(graph_data)
    graph.render(plot_title, view = True)
    return graph
tree_graph = plot_tree(tree, df, 'Variety', le, 'Iris')
tree_graph


# In[9]:


from IPython.display import Image
Image("C:\\naive_bayes_data.png")


# In[10]:



# Assigning features and label variables
weather=['Rainy','Rainy','Overcast','Sunny','Sunny','Sunny',
         'Overcast','Rainy','Rainy','Sunny','Rainy','Overcast',
         'Overcast','Sunny']

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool'
      'Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# In[11]:


import pandas as pd
data= {'weather': ['Rainy','Rainy','Overcast','Sunny','Sunny','Sunny',
                   'Overcast','Rainy','Rainy','Sunny','Rainy','Overcast',
                   'Overcast','Sunny'],
      'temp': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
      'play': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']}

data= pd.DataFrame(data)
data


# In[12]:


wheather_encoded=le.fit_transform(data['weather'])
print (wheather_encoded)


# In[13]:


temp_encoded=le.fit_transform(data['temp'])
label=le.fit_transform(data['play'])
print ("Weather:",wheather_encoded)
print ("Temp:",temp_encoded)
print ("Play:",label)


# In[14]:


features=zip(wheather_encoded,temp_encoded)
features_ls= list(features)
print(features_ls)


# In[15]:


from sklearn.naive_bayes import GaussianNB, BernoulliNB

model = BernoulliNB()

model.fit(features_ls,label)

predicted= model.predict([[0,2]]) 
print ("Predicted Value:", predicted)


# Exercise 2.3: Repeat the experiment again but now using Gaussian Naïve Bayes (just replace model= GaussianNB())! Is the result the same?
# 
# YES 

# In[16]:


model = GaussianNB()

model.fit(features_ls,label)

predicted= model.predict([[0,2]]) 
print ("Predicted Value:", predicted)

