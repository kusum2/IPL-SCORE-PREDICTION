#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


ipl = pd.read_csv('ipl2017.csv')


# In[3]:


X = ipl.iloc[:,[7,8,9,12,13]].values 
y = ipl.iloc[:, 14].values


# In[4]:


ipl.shape


# In[5]:


ipl.info()


# In[6]:


ipl.isnull().sum()


# In[7]:


ipl.head()


# In[8]:


sns.pairplot(ipl[['runs','wickets','overs','striker','non-striker']])


# In[9]:


ipl.drop(['date','runs_last_5','wickets_last_5'], axis=1,inplace=True)


# In[10]:


ipl.head()


# In[11]:


ipl.dtypes


# In[12]:


type(ipl)


# In[13]:


ipl=ipl.iloc[:,:].values


# In[14]:


ipl


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[16]:


match=LabelEncoder()


# In[17]:


ipl[:,1]=match.fit_transform(ipl[:,1])


# In[18]:


ipl[:,2]=match.fit_transform(ipl[:,2])


# In[19]:


ipl


# In[20]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[21]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[1])],remainder='passthrough')


# In[22]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[2])],remainder='passthrough')


# In[23]:


b=ct.fit_transform(ipl)


# In[24]:


b


# In[25]:


b=pd.DataFrame(b)
b


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# In[27]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # RandomForestRegressor

# In[33]:


from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=500,random_state=0)
reg.fit(X_train,y_train)


# In[34]:


y_pred = reg.predict(X_test)
score = reg.score(X_test,y_test)*100


# In[35]:


y_pred


# In[36]:


score


# In[ ]:




