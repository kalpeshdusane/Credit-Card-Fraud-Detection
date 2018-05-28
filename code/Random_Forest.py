
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
df = pd.read_csv('creditcard.csv', low_memory=False)
df.head()
x = df.iloc[:,:-1]
y = df['Class']
# x.head()
frauds = df.loc[df['Class'] == 1]
non_frauds = df.loc[df['Class'] == 0]
print("We have", len(frauds), "fraud data points and", len(non_frauds), "regular data points.")


# In[26]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.3, 
                                                    random_state=42)
print("Size of training set: ", X_train.shape)


# In[27]:


scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)


# In[28]:


# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train[0,:]


# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


# clf = RandomForestClassifier(max_depth=100, random_state=10)
clf = RandomForestClassifier(random_state=0)
# clf = RandomForestClassifier()


# In[31]:


clf.fit(X_train,y_train)


# In[32]:


print(clf.feature_importances_)


# In[33]:


predictions = clf.predict(X_test)
print("Size of training set: ", X_test.shape)
print(predictions.shape)


# In[34]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[35]:


print(classification_report(y_test,predictions))


# # after Oversampling

# In[36]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas_ml as pdml
import imblearn

x = scale(x)
X2_train, X2_test, y2_train, y2_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(X2_train.shape)
print(X2_test.shape)


# In[37]:


print(df.shape)
df2 = pdml.ModelFrame(X2_train, target=y2_train)
print(df2.shape)
sampler = df2.imbalance.over_sampling.SMOTE()

oversampled = df2.fit_sample(sampler)
print(oversampled.shape)
# print(oversampled.iloc[:,:-1])
y2_train = oversampled['Class']
y2_train = y2_train.as_matrix()

del oversampled['Class']

X2_train = oversampled.iloc[:,:]
print(X2_train)


# In[38]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)


# In[39]:


clf.fit(X2_train,y2_train)


# In[40]:


print(clf.feature_importances_)


# In[41]:


predictions = clf.predict(X2_test)
print("Size of training set: ", X2_test.shape)
print(predictions.shape)


# In[42]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y2_test,predictions))


# In[43]:


print(classification_report(y2_test,predictions))


# In[ ]:




