#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
from azureml.core import Workspace, Run
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace, Dataset
import os
import joblib


# In[2]:


ia = InteractiveLoginAuthentication(tenant_id='e4e34038-ea1f-4882-b6e8-ccd776459ca0')
ws = Workspace(subscription_id= "c59b6c0a-0bc0-4b69-bd03-020b2171f742",
    resource_group="RG-AmlWS-DSTeam-RnD",
    workspace_name= "aml-DSTeam-RnD-001", auth=ia)
print(f'worspace details {ws}')


# In[3]:


run = Run.get_context()


# In[4]:


df=pd.read_csv('car data.csv')


# In[5]:


final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]


# In[6]:


final_dataset['Current Year']=2020


# In[7]:


final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']


# In[8]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[9]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[10]:


final_dataset=final_dataset.drop(['Current Year'],axis=1)


# In[11]:


X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[12]:


### Feature Importance

# from sklearn.ensemble import ExtraTreesRegressor
# import matplotlib.pyplot as plt
# model = ExtraTreesRegressor()
# model.fit(X,y)


# In[13]:


# print(model.feature_importances_)


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[15]:


from sklearn.ensemble import RandomForestRegressor


# In[21]:


# regressor=RandomForestRegressor()


# In[22]:


# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# print(n_estimators)


# In[28]:


# from sklearn.model_selection import RandomizedSearchCV


# In[24]:


#Randomized Search CV

# Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 5, 10]


# In[31]:


# Create the random grid
# random_grid = {'n_estimators': 100,
#                'max_features': 'sqrt',
#                'max_depth': 5,
#                'min_samples_split': 2,
#                'min_samples_leaf': 1}

# print(random_grid)


# In[32]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor(n_estimators=100, random_state=0)


# In[33]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, verbose=2, random_state=42, n_jobs = 1)


# In[34]:


model = rf.fit(X_train,y_train)


# In[35]:


predictions=model.predict(X_test)


# In[36]:


from sklearn import metrics


# In[38]:


import numpy as np
mae = metrics.mean_absolute_error(y_test, predictions)
run.log('MAE',np.float(mae))


# In[ ]:


mse =  metrics.mean_squared_error(y_test, predictions)
run.log('MSE',np.float(mse))


# In[39]:


rmse =  np.sqrt(metrics.mean_squared_error(y_test, predictions))
run.log('RMSE',np.float(rmse))


# In[44]:


os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/random_forest_regression_model.pkl')


# In[136]:





# In[ ]:


run.complete()

