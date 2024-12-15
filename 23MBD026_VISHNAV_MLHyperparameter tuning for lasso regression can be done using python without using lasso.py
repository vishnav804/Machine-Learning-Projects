#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Aim:Hyperparameter tuning for lasso regression can be done using python without using lassoCv_api
import pandas as pd
df=pd.read_csv("BostonHousing.csv")
df.head()


# In[10]:


x=df.iloc[:,:-1]
print(x.shape)


# In[8]:


y=df.iloc[:,-1]
y.shape


# In[9]:


from sklearn.linear_model import Lasso
model=Lasso()


# In[11]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)
model.fit(xtrain,ytrain)


# In[14]:


from sklearn.model_selection import RepeatedKFold
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)


# In[15]:


from sklearn.metrics import r2_score
ypred=model.predict(xtest)
r2_score(ytest,ypred)


# In[21]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_sc=sc.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(x_sc,y,test_size=0.25,random_state=1)
model1=Lasso()
parms={'alpha':[0.00001,0.0001,0.001,0.01]}
from sklearn.model_selection import GridSearchCV
search=GridSearchCV(model1,parms,cv=cv)
result=search.fit(x_sc,y)
result.best_params_
    


# In[22]:


model2=Lasso(alpha=0.01)
model2.fit(xtrain,ytrain)


# In[24]:


ypred2=model2.predict(xtest)
r2_score(ytest,ypred2)


# In[ ]:




