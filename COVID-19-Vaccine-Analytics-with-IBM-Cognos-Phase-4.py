#!/usr/bin/env python
# coding: utf-8

# # Module 7 Data Exploration & Visualization

# In[8]:


import numpy as np
import pandas as pd
import tubesml as tml
from tubesml.base import BaseTransformer, reset_columns, self_columns
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import os

# List files in the specified directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read the data from a CSV file
data = pd.read_csv('country_vaccinations.csv')

# Display the first few rows of the data
data.head()


# In[ ]:


# data.info()


# # Data Exploration

# In[10]:


train.hist(bins=50, figsize=(20,15), grid=False)
plt.show()


# In[13]:


_ = tml.plot_correlations(train)


# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have 'res' as a DataFrame
# Define and assign 'res'
data = {'param_model__min_samples_leaf': [10, 10, 10],
        'param_proc__feats__waist_ratio': [False, False, False],
        'param_proc__feats__weight_ratio': [False, False, False],
        'param_model__max_features': ['sqrt', 'sqrt', 'sqrt'],
        'mean_train_score': [0.8, 0.85, 0.75],
        'mean_test_score': [0.7, 0.75, 0.65]}

res = pd.DataFrame(data)

# Create subplots
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Your code to plot using 'res'
# ...

plt.show()


# In[36]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load the data from 'country_vaccinations.csv'
df = pd.read_csv('country_vaccinations.csv')

# Display the first few rows of the DataFrame to inspect the data
print(df.head())

# You can also use df.info() to get information about the data types and missing values
# print(df.info())


# In[37]:


df.head()


# In[38]:


df.shape  # Dataset Shape


# In[39]:


df.info() # Dataset Information


# In[40]:


df.describe()   # Statistics


# In[41]:


df.isnull().sum() 


# In[ ]:


# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define 'y_test' and 'y_pred' with actual and predicted values
y_test = [1, 2, 3, 4, 5]  # Replace with your actual values
y_pred = [1.2, 2.3, 2.9, 3.7, 4.5]  # Replace with your predicted values

plt.figure(figsize=(10, 6))

# Create a scatterplot
sns.scatterplot(x=y_test, y=y_pred, color='b', alpha=0.9, edgecolor='k', s=80)

# Add a regression line
sns.regplot(x=y_test, y=y_pred, scatter=False, color='r', line_kws={"color": "orange", "lw": 2})

plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.title("Model Performance - Actual vs. Predicted Values", fontsize=16)

plt.show()


# 

# 

# 

# In[ ]:




