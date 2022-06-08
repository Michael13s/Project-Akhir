#!/usr/bin/env python
# coding: utf-8

# ### Impott lib

# In[34]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# ### lOAD DATASET and explpr

# In[35]:


df = pd.read_csv("weight-height.csv")


# In[36]:


df.head()


# In[37]:


df.shape


# In[38]:


df.dtypes


# In[39]:


df.count()


# ### Missing Value

# In[40]:


df.isnull().sum()


# ### Scaling and labeling

# In[41]:


X = df.iloc[:, :-1].values
y = df.iloc[:, 2].values


# In[53]:


from sklearn.preprocessing import LabelEncoder
std_scaler = StandardScaler()
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

import numpy as np
X = np.vstack(X[:, :]).astype(np.float)


# ### rubah value jenis kelamin

# In[43]:


df['Gender'].replace('Female',0, inplace=True)
df['Gender'].replace('Male',1, inplace=True)
X = df.iloc[:, :-1].values
y = df.iloc[:, 2].values


# ### Split & train dataset

# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ### Masukan Model

# In[49]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
model = lin_reg.fit(X_train, y_train)


# In[50]:


accuracy = model.score(X_test, y_test)


# In[51]:


print(f"Akurasi Model: {accuracy * 100}%")


# ### Menyimpan Model dan scalar

# In[54]:


joblib.dump((model, std_scaler), "hegiht-weight-prediction-LR.pkl")


# In[ ]:




