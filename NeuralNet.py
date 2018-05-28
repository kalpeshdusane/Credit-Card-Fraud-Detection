
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import sklearn
# import tensorflow as tf


# In[4]:


df = pd.read_csv('creditcard.csv', low_memory=False)
df.head()
x = df.iloc[:,:-1]
y = df['Class']
# x.head()
frauds = df.loc[df['Class'] == 1]
non_frauds = df.loc[df['Class'] == 0]
print("We have", len(frauds), "fraud data points and", len(non_frauds), "regular data points.")


# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.3, 
                                                    random_state=42)
print("Size of training set: ", X_train.shape)


# # Simple Neural Network

# In[11]:


scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)


# In[12]:


# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train[0,:]


# In[13]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,200,30)
#                     ,max_iter=100
                   )


# In[14]:


mlp.fit(X_train,y_train)


# In[15]:


predictions = mlp.predict(X_test)
print("Size of training set: ", X_test.shape)
print(predictions.shape)


# In[16]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[17]:


print(classification_report(y_test,predictions))


# # Neural Network after Oversampling

# In[58]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas_ml as pdml
import imblearn


# In[59]:


print(df.shape)
df2 = pdml.ModelFrame(x, target=y)
print(df2.shape)
sampler = df2.imbalance.over_sampling.SMOTE()

oversampled = df2.fit_sample(sampler)
print(oversampled.shape)
X2, y2 = oversampled.iloc[:,:-1], oversampled['Class']


# In[60]:


data = scale(X2)
pca = PCA(n_components=10)
X2 = pca.fit_transform(data)
# X2 = scale(X2)
X2


# In[61]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2.as_matrix(), test_size=0.3, random_state=42)
print(X2_train.shape)
print(X2_test.shape)


# In[62]:


mlp = MLPClassifier(hidden_layer_sizes=(30,200,30)
#                     ,max_iter=100
                   )


# In[63]:


mlp.fit(X2_train,y2_train)


# In[64]:


predictions = mlp.predict(X2_test)
print("Size of training set: ", X2_test.shape)
print(predictions.shape)


# In[65]:


# CM = confusion_matrix(y2_test,predictions)
print(confusion_matrix(y2_test,predictions))
# CM.print_stats()


# In[66]:


print(classification_report(y2_test,predictions))


# In[67]:


from sklearn.metrics import accuracy_score
accuracy_score(y2_test, predictions)

