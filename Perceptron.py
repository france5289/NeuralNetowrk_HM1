
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#get_ipython().magic('matplotlib inline')


# In[2]:

iris = pd.read_csv('iris.data', names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name'])


# In[3]:

test_data=iris[iris['Name'].isin(['Iris-setosa', 'Iris-versicolor'])] 
# select 'Iris-setosa' and 'Iris-versicolor' data to form testing dataset as pd.dataframe format
test_data.head()


# In[4]:

test_data['Class'] = np.where(test_data['Name']=='Iris-setosa', 1,-1)


# In[5]:

y = test_data['Class'].to_numpy()
X = test_data.drop(columns=['Name', 'Class']).to_numpy()
# bias = 1
X = np.insert(X, 0, 1, axis = 1)
X


# In[6]:

y


# In[30]:

# the perceptron alogorithm implementation
def Perceptron(X,y, l_rate):
    w_history = [] # weight history
    w = np.array([-1, 5, 3, 2, 1]) # weight initialization w0 = threshhold
    w_old = np.ones(5)
    t = 0
    def sgn(v):
        if v >= 0:
            return 1
        else:
            return -1
    
    while np.linalg.norm(w-w_old) > 0.1:
        w_old = np.copy(w)
        count = 0 # counter for correctness
        for i, x in enumerate(X):
            #print("i: ", i)
            #print("x: ", x)
            # calculate the inner product between w and x
            v = np.dot(w,x)
            # pass the result of inner product to sgn() function
            sgn_v = sgn(v)
            if y[i] == 1 and sgn_v == -1:
                count = 0
                w = w + l_rate*x
                print('Wrong Classification! {a} should be C1 but classified to C2'.format(a=x))
            elif y[i] == -1 and sgn_v == 1:
                count = 0
                w = w - l_rate*x
                print('Wrong Classification! {a} should be C2 but classified to C1'.format(a=x))
            else:
                count = count + 1
                print('Correct Classification! No.{a} x = {b}'.format(a = count, b = x))
            print('Current Weight:', w)
            print('-----------------------------------------------------------------------')
            #print('{a:2d} iteration, w={b}'.format(a=t,b=w))
            t += 1
            w_history.append(w.copy())
        print('######################################################################')
        print('End of this round, we have find a temporary weight')
        print('Start another round to check whether this weight is correct or not!')
        print('######################################################################')
        input('Please press any key to start a new round!')
    print('Congratulation! We have found an optimal weight!')
    #return w, w_history
            


# In[31]:

Perceptron(X,y, l_rate = 0.2)


# In[ ]:




# In[ ]:




# 

# In[ ]:



