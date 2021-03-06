
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn


# In[2]:


df = pd.read_csv('creditcard.csv', low_memory=False)
df.head()
x = df.iloc[:,:-1]
y = df['Class']
# x.head()
frauds = df.loc[df['Class'] == 1]
non_frauds = df.loc[df['Class'] == 0]
print("We have", len(frauds), "fraud data points and", len(non_frauds), "regular data points.")


# In[3]:


from sklearn.preprocessing import scale
x = scale(x)


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.3, 
                                                    random_state=42)
print("Size of training set: ", X_train.shape)


# In[5]:


from sklearn import linear_model


# In[6]:


logreg = linear_model.LogisticRegression()


# In[7]:


logreg.fit(X_train, y_train)


# In[9]:


predictions = logreg.predict(X_test)
print("Size of training set: ", X_test.shape)
print(predictions.shape)


# In[10]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[11]:


print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# # after Oversampling

# In[12]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas_ml as pdml
import imblearn


# In[16]:


X2_train, X2_test, y2_train, y2_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(X2_train.shape)
print(X2_test.shape)


# In[17]:


print(df.shape)
df2 = pdml.ModelFrame(X2_train, target=y2_train)
print(df2.shape)
sampler = df2.imbalance.over_sampling.SMOTE()

oversampled = df2.fit_sample(sampler)
print(oversampled.shape)
# print(oversampled.iloc[:,:-1])
y2_train = oversampled['Class']

del oversampled['Class']

X2_train = oversampled.iloc[:,:]
# print(X2_train)


y2_train = y2_train.as_matrix()
X2_train = X2_train.as_matrix()
type(X2_train)


# In[18]:


from sklearn import linear_model


# In[19]:


logreg = linear_model.LogisticRegression()


# In[20]:


logreg.fit(X2_train, y2_train)


# In[21]:


predictions = logreg.predict(X2_test)
print("Size of training set: ", X2_test.shape)
print(predictions.shape)


# In[22]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y2_test,predictions))


# In[23]:


print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[ ]:




