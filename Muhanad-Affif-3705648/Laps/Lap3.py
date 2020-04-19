#!/usr/bin/env python
# coding: utf-8

# In[3]:


def conform(fav):
    fav=42
    return fav
if __name__=="__main__":
    print('welcome!')
    fav=7
    conform(fav)
    print("my favorite # is" , fav)


# In[4]:


student={'A':28,'B':25,'C':32,'D':25}
def test (student):
    new={'E':30,'F':28}
    student.update(new)
    print("inside the function" , student)
    return
test(student)
print("outside the function" ,student)


# In[5]:


class student:
    def __init__(self,name):
        self.name=name
        self.course_list=[]
std=student("set_her_your_name")
print(std.name)


# In[6]:


class student:
    def __init__(self,name):
        self.name=name
        self.course_list=[]
    def add(self , new_course):
        self.course_list.append(new_course)
std=student("set_her_your_name")
std.add("Paython")
print(std.course_list)


# In[ ]:


class student:
    def __init__(self,name):
        self.name=name
        self.course_list=[]
        
    def add(self , new_course):
        self.course_list.append(new_course)
        
std=student("set_her_your_name")
txt = input("type somthing to test this out:")
std.add(txt)
print(std.course_list)

while True:
    course_list = input('Enter your course name to add it:'+
                   '\n (hint: to stop adding the courses, do not type anything)'+
                   '\n')
    
    if course_list != '':
        std.add(course_list)
        print('the course added succesfully')
    else:
        break;
        
print(std.name)
print(std.courses_list)

while True:
    course_list = input('Enter your course name to delete it:'+
                   '\n (hint: to stop deleting the courses, do not type anything)'+
                   '\n')
    
    if course_list != '':
        if course_list in std.courses_list:
            std.delete(course_list)
            print('the course deleted succesfully')
        else:
            print('cannot find this course')
    else:
        break;

print(std.courses_list)


# In[ ]:


class person:
    def __init__(self,fname,lname):
        self.firstname=fname
        self.lastname=lname
        
    def printname(self):
        print(self.firstname ,self.lastname)
        
class professor(person):
    pass
mhd = professor("mohammed" ,"alsarem")
mhd.printname()


# In[ ]:





# In[ ]:


class Person:
    def __init__(self, fname, lname):
        self.fname = fname
        self.lname = lname
    
    def printname(self):
        print(self.fname, self.lname)
    
class Student(Person):
    pass
    def __init__(self, fname, lname):
        self.fname = fname
        self.lname = lname

std = Student('Muhannad', 'Afifi')
std.printname()


# In[ ]:


class Student(Person):
    pass
    age = 0
    def __init__(self, fname, lname, age):
        super(Student, self).__init__(fname, lname)
        self.age = age

std = Student('Muhannad', 'Afifi', 25)
std.printname()
print(std.age)


# In[ ]:




