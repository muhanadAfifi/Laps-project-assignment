#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mlxtend


# In[2]:


from mlxtend.frequent_patterns import apriori


# In[5]:


dataset = [['Drink', 'Nuts', 'Diaper'],
           ['Drink', 'Coffee', 'Diaper'],
           ['Drink', 'Diaper', 'Eggs'],
           ['Nuts', 'Eggs', 'Milk'],
           ['Nuts', 'Coffee', 'Diaper', 'Eggs', 'Milk']]
dataset


# In[7]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

TranEncod = TransactionEncoder()
te_ary = TranEncod.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=TranEncod.columns_)

df


# In[8]:


apriori(df, min_support=0.6)


# In[9]:


apriori(df, min_support=0.6, use_colnames=True)


# In[10]:


apriori(df, min_support=0.3, use_colnames=True)


# In[11]:


frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# In[12]:


frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.5) ]


# In[13]:


frequent_itemsets[ frequent_itemsets['itemsets'] == {'Diaper', 'Drink'}]

