#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


housing = pd.read_csv('/Users/user/UniIDE/USWS/Datasets/housing.csv')


# In[7]:


housing.shape
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()


# In[8]:


housing.describe()


# In[9]:


housing.hist(bins=50, figsize=(20,15))


# In[14]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, s=housing["population"]/100, label="population", c="median_house_value", cmap=plt.get_cmap("jet"))


# In[16]:


corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[20]:


housing_na=housing.dropna(subset=["total_bedrooms"])
housing_na.shape


# In[25]:


dummies=pd.get_dummies(housing_na.ocean_proximity)
housing_na_dummies=pd.concat([housing_na, dummies], axis='columns')
housing_na_dummies.head()
housing_clean=housing_na_dummies.drop(['ocean_proximity', 'ISLAND'],axis='columns')
housing_clean.head()


# In[28]:


# Create features and label datasets
X=housing_clean.drop(columns='median_house_value')
X.head()
y=housing_clean['median_house_value']
y.head()


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1984)


# In[30]:


from sklearn.linear_model import LinearRegression
OLS = LinearRegression()

OLS.fit(X_train,y_train)


# In[31]:


# display the intercept and coefficients of the OLS model
print("Intercept is: " + str(OLS.intercept_))
print ("The set of coefficients are " + str(OLS.coef_))
print("The R-squared value is: " + str(OLS.score(X_train, y_train)))


# In[36]:


# predicting with OLS
y_pred=OLS.predict(X_test)
performance=pd.DataFrame({'PREDICTIONS':y_pred, 'ACTUAL VALUES':y_test})
performance['error']=performance['ACTUAL VALUES']-performance['PREDICTIONS']
performance.head()


# In[37]:


# preparing data for plotting 

performance.reset_index(drop=True, inplace=True)
performance.reset_index(inplace=True)
performance.head()


# In[41]:


# plotting residuals

fig = plt.figure(figsize=(10,5))
plt.bar('index', 'error', data=performance[:50], color='black', width=0.3)
plt.xlabel("Observations")
plt.ylabel("Residuals")
plt.show()


# In[44]:


import statsmodels.api as sm
X_train=sm.add_constant(X_train)
nicerOLS=sm.OLS(y_train, X_train).fit()
nicerOLS.summary()


# In[ ]:




