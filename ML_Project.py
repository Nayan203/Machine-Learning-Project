#!/usr/bin/env python
# coding: utf-8

# There is an Ad Agencies which uses different modes of meduim for the advertisement like TV, NewsPaper & Radio. And they have sales for the product.
# 
# Aim of Project: To identify, In which medium to put more amount of money for advertisement so that their sales can be increased in next 1 year or 1 month. 

# In[ ]:


#import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# importing advertising data set

adv = pd.read_csv('advertising.csv')


# In[3]:


adv.info()


# In[4]:


adv.head()


# In[5]:


adv.isnull().sum()


# In[6]:


adv.describe()


# In[7]:


plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
plt.title('TV')
plt.scatter(adv['Sales'],adv['TV'])
plt.subplot(1,3,2)
plt.title('Radio')
plt.scatter(adv['Sales'],adv['Radio'])
plt.subplot(1,3,3)
plt.title('Newspaper')
plt.scatter(adv['Sales'],adv['Newspaper'])
plt.show()


# In[8]:


adv.corr()


# In[9]:


#Lets built simple linear regression with TV and Sales

X = adv['TV']
y = adv['Sales']


# In[10]:


#test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[11]:


X_train.head()


# In[12]:


X_test.head()


# In[13]:


#Performing the linear regression using statsmodel
import statsmodels.api as sm


# In[14]:


#Statsmodel fits the line passing through the origin hence we add constant for the intercept

X_train_sm = sm.add_constant(X_train)


# In[15]:


lr = sm.OLS(y_train,X_train_sm).fit()


# In[16]:


lr.params


# In[ ]:


Sales = 0.05*TV + 6.95


# In[17]:


lr.summary()


# In[18]:


#plotting the scatter graph
plt.scatter(X_train,y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[19]:


#calculating the residues
y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[20]:


#scatter plot
plt.scatter(X_train,res)


# In[21]:


#error term distribution
sns.distplot(res,bins=15)
plt.show()


# In[22]:


#Predicting it on the test data
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[23]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[24]:


#checking the accuracy
np.sqrt(mean_squared_error(y_test, y_pred)) #rmse root mean square error


# In[25]:


r2_score(y_test,y_pred)


# #Is it better to scale the variables..
# Yes, it would be better so that all variables will be in the same scale.
# 
# #There are two kinds of scaling
# Standard scaling --> mean is 0 and std.dev is 1
# min-max scaling  --> all the values will be b/w 0 and 1

# In[26]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[27]:


scaler = MinMaxScaler()


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[29]:


X_train.head()
X_train.shape
type(X_train)


# In[30]:


X_train_scaled = np.array(X_train)


# In[31]:


X_train_scaled1 = X_train_scaled.reshape(-1,1)


# In[32]:


X_train_scaled2 = pd.DataFrame(X_train_scaled1)


# In[33]:


y_train_scaled=np.array(y_train)
y_train_scaled1 = y_train_scaled.reshape(-1,1)
y_train_scaled2 = pd.DataFrame(y_train_scaled1)


# In[34]:


X_train_scaled3 = scaler.fit_transform(X_train_scaled2)
y_train_scaled3 = scaler.fit_transform(y_train_scaled2)


# In[35]:


# Let's fit the regression line following exactly the same steps as done before
X_train_scaled4 = sm.add_constant(X_train_scaled3)

lr_scaled = sm.OLS(y_train_scaled3,X_train_scaled4).fit()


# In[36]:


lr_scaled.params


# In[37]:


lr_scaled.summary()

