#!/usr/bin/env python
# coding: utf-8

# In[43]:


from platform import python_version
print(python_version())


# In[68]:


# filename.py
if __name__=="__main__":
    pass


# In[67]:


if __name__=="__main__":
    print("Hello,world")


# In[66]:


x = 5
y= 2**2 +1
x==y
x,y=5, 2**(2+1)
x==y


# In[72]:


def add(x,y):
    return x+y
add(2,5)


# In[ ]:





# In[71]:


def add(x,y):
    return x, y , x + y
FV , SV , S = add(2,5)
print("First variable is {} Secound variable is {} And the summation is {}".format(FV,SV,S))


# In[5]:


from numpy import pi
def sphere_volume(r):
    return (4/3) * pi * (r ** 3)

sphere_volume(3)


# In[3]:


def information(Course , Professor , Level='8'):
    print(Course + 'is in' + Level + 'which is delivered by' + Professor)
    
if __name__=='__main__':
    Course= 'IS-372 Data Mining & Data Warehouse'
    Professor= 'A/Proof. Mohammed Al-Sarem'
    information(Course,Professor)


# In[51]:


Course_Name='IS-372 Data Mining & Data Warehouse'
print(Course_Name[2])


# In[4]:


course_name = "IS-372 Data Mining & Data Warehouse"
print(course_name[:6]) #from index 0 to 6
print(course_name[-1]) #last character
print(course_name[:5]) #from index 0 to 5
print(course_name[6:])


# In[6]:


my_list = ['Hello', 93.8, 'world', 10]


print(my_list[0])

print(my_list[-2])

print(my_list[:2])


# In[52]:


my_list = [1,2] 
my_list.append(4)
my_list.insert(2,3)
my_list.remove(3)
my_list.pop()


# In[53]:


def arithmetic(a,b):
    return a - b , a*b
x,y = arithmetic(5,2)
print(x,y)
both = arithmetic(5,2)
print(both)


# In[7]:


def list_ops():
    list = ['bear', 'ant', 'cat', 'dog']
    # 1
    list.append("eagle")
    # 2
    list.pop(2)
    list.insert(2, "fox")
    # 3
    list.pop(1)
    # 4
    list.reverse()
    # 5
    list.insert(list.index("eagle"), "hawk")
    list.remove("eagle")
    # 6
    list.append("hunter")
    
    return list

print(list_ops())


# In[8]:


DataMining_Professores = {'Muhannad, Al-Mohaeemid', 'Faisal, Saeed', 'Mohammed, Al-Sarem'}
print(DataMining_Professores)

DataMining_Professores.add('Wadii, Boulila') #add 
DataMining_Professores.discard('Muhannad, Al-Mohaeemid') #remove
print(DataMining_Professores)

Database_Professores = {'Muhannad, Al-Mohaeemid', 'Faisal, Saeed','Essa, Hizzam', 'Wadii, Boulila'}

DataMining_Professores.intersection(Database_Professores)
DataMining_Professores.difference(Database_Professores)


# In[9]:


DataScience_Track = {'IS-372':'Data Mining & Data Warehouse',
                     'IS-472':'Decision Support System',
                     'IS-476':'Information Searxh, Retrieval & Visualization',
                     'IS-453':'Special Topice in Data Mangement'}

print(DataScience_Track['IS-453'])

print(DataScience_Track.keys())

print(DataScience_Track.values())


# In[73]:


if(2>3):
    print("W")
elif(3>4):
    print("N")
else:
    print("Muhanad")


# In[10]:


i=0
while(i <10):
    print(i , end=" ")
    i+=1


# In[11]:


colors = ["Red","Orange","Blue"]
for n in colors:
    print(n + "?") 


# In[12]:


s = "stab"
for i in range(len(s)):  
    print (s[0 : i : 1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




