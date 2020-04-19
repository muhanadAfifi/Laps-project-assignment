#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
a = np.arange(15).reshape(3,5)
a


# In[18]:


print(np.array([8,4,6,0,2]))


# In[ ]:





# In[19]:


print('create a 2-D array by passing a list of lists into array().')
A = np.array([[1,2,3],[4,5,6]])
print(A)

print('access elements of the array with brackets.')
print(A[0,1],A[1,2])

print('the elements of 2-D array are 1-D arrays.')
print(A[0])


# In[ ]:





# In[20]:


def example1():
    A = np.array([[3, -1, 4],
                  [1, 5, -9]])
    B = np.array([[2, 4, -5, 6],
                  [-1, 7, 9, 3],
                  [3, 2, -7, -2]])
    return np.dot(A, B)

example1()


# In[19]:


A = np.array([[3,-1,4],[1,5,-9]])
B = np.array([[2,4,-5,6],[-1,7,9,3],[3,2,-7,-2]])
print('arrays' , 'A=',A ,'B=',B,'return the matrix product')

np.dot(A,B)


# In[22]:


arr = np.ndarray(shape = (5,1), dtype='int64')
arr = arr + 5
print(arr)


# In[23]:


print('addition concatenates lists togather')
print([1,2,3] + [4,5,6])

print('mutliplication cocatenates a list with itself a given number of times')
print([1,2,3] * 4)


# In[24]:


x = np.array([3, -4, 1])
y = np.array([5, 2, 3])

print(x + 10)
print(y * 4) 
print(x + y) 
print(x * y) 


# In[25]:


a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.dtype)
print(a.ndim)
print(a.shape)
print(a.size)
print(a[1,2])


# In[27]:


x = np.arange(10)
print(x)

print(x[3]) # slicing index 3
print(x[:4]) # slicing from index 0 to 4
print(x[4:]) # slicing from index 4 to the last index
print(x[4:8]) # slicing from index 4 to 8 


# In[30]:


x= np.array([[0,1,2,3,4],
             [5,6,7,8,9]])
print(x[1, 2]) 
print(x[:,2:])


# In[31]:


x = np.arange(0, 50, 10)
print(x)
index = np.array([3, 1, 4])
print(x[index])

# A boolean array extracts the elements of 'x' at the same places as 'True'
mask = np.array([True, False, False, True, False])
print(x[mask])


# In[32]:


y =np.arange(10,20,2)
print(y)
mask = y > 15
print(mask)
print(y[mask])

y[mask] = 100
print(y)


# In[33]:


from sklearn import datasets
iris = datasets.load_iris()
print(iris.filename)


# In[42]:


import numpy as np
iris_data = np.genfromtxt('C:\Users\W\Anaconda3\lib\site-packages\sklearn\datasets\data\iris.csv',
                     delimiter=",", skip_header=1)

print(iris_data)


# In[43]:


print('mean of {} is {}'.format(iris.feature_names[0], data[:,0].mean()))


# In[44]:


print('mean of {} is {}'.format(iris.feature_names[0], data[:,0].mean()))
print('std of {} is {}'.format(iris.feature_names[0], data[:,0].std()))
print('var of {} is {}'.format(iris.feature_names[0], data[:,0].var()))
print('max of {} is {}'.format(iris.feature_names[0], data[:,0].max()))
print('min of {} is {}'.format(iris.feature_names[0], data[:,0].min()))


# In[ ]:





# In[ ]:




