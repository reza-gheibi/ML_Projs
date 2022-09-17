#!/usr/bin/env python
# coding: utf-8

# # Question 3
# 

# In[60]:


import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier


# In[61]:


#reading the data 
data_frame = pd.read_csv("Data_for_UCI_named.csv")
data_frame.head()


# In[62]:


# 3-(a)
column_index = [4,12]
final_dataframe = data_frame.drop(data_frame.columns[column_index],axis=1)
final_dataframe.head()


# In[63]:


# 3-(b)
final_dataframe['stabf']=final_dataframe['stabf'].replace('stable',1)
final_dataframe['stabf']=final_dataframe['stabf'].replace('unstable',0)
final_dataframe.head()


# In[64]:


# 3-(c)
test_percent = 20
total_rows = final_dataframe.shape[0] 
test_subset=(test_percent*total_rows//100)
test = final_dataframe.iloc[0:test_subset]


# In[65]:


# 3-(d)
train1 = final_dataframe.iloc[test_subset:]
train_percent = 75
total_rows2 = train1.shape[0]
train_subset=(train_percent*total_rows2//100)
train = train1.iloc[0:train_subset]
valid = train1.iloc[train_subset:]


# # Question 4

# In[66]:


# 4-(a)
train_predictor = train.iloc[:,0:11]
train_target = train.iloc[:,11]
gini_tree = DecisionTreeClassifier(criterion='gini', max_depth = 5)
gini_model=gini_tree.fit(train_predictor,train_target)


# In[67]:


# 4-(b)
valid_predictor = valid.iloc[:,0:11]
valid_target = valid.iloc[:,11]
predictions=gini_model.predict_proba(valid_predictor)
entropy1 = log_loss(valid_target,predictions)
print(entropy1)


# In[68]:


# 4-(c)
info_gain_tree = DecisionTreeClassifier(criterion='entropy', max_depth = 5)
info_gain_model=info_gain_tree.fit(train_predictor,train_target)


# In[69]:


# 4-(d)
predictions2=info_gain_model.predict_proba(valid_predictor)
entropy2 = log_loss(valid_target,predictions2)
print(entropy2)


# In[70]:


# 4-(e)
# Model with Gini performed better due to lower value of entropy

train_predictor = train1.iloc[:,0:11]
train_target = train1.iloc[:,11]
gini_tree = DecisionTreeClassifier(criterion='gini', max_depth = 5)
gini_model=gini_tree.fit(train_predictor,train_target)


test_predictor = test.iloc[:,0:11]
test_target = test.iloc[:,11]
predictions=gini_model.predict_proba(test_predictor)
entropy3 = log_loss(test_target,predictions)
print(entropy3)


# In[71]:


# gini performed better 

