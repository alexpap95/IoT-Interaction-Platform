#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python preprocess_data.py -i data.csv -o dataset.data')


# In[2]:


import numpy as np
import cPickle as cp


# ### Load the sensor data
Load the OPPORTUNITY processed dataset. Sensor data is segmented using a sliding window of fixed length. The class associated with each segment corresponds to the gesture which has been observed during that interval. Given a sliding window of length T, we choose the class of the sequence as the label at t=T, or in other words, the label of last sample in the window.
# In[3]:


def load_dataset(filename):

    f = file(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


# In[4]:


print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('dataset.data')


# In[5]:


first_conc_x = np.concatenate(X_train[0:15, :])
for i in range(15, X_train.shape[0], 15):
    first_conc_x = np.vstack((first_conc_x, np.concatenate(X_train[i:(i+15), :])))
X_train_kat = first_conc_x


# In[6]:


first_conc_y = np.unique(y_train[0:15])[0]
for i in range(15, y_train.shape[0], 15):
    first_conc_y = np.vstack((first_conc_y, np.unique(y_train[i:(i+15)])[0]))
y_train_kat = first_conc_y


# In[7]:


first_conc_x = np.concatenate(X_test[0:15, :])
for i in range(15, X_test.shape[0], 15):
    first_conc_x = np.vstack((first_conc_x, np.concatenate(X_test[i:(i+15), :])))
X_test_kat = first_conc_x


# In[8]:


first_conc_y = np.unique(y_test[0:15])[0]
for i in range(15, y_test.shape[0], 15):
    first_conc_y = np.vstack((first_conc_y, np.unique(y_test[i:(i+15)])[0]))
y_test_kat = first_conc_y


# In[9]:


y_train_kat.shape[0] == X_train_kat.shape[0]


# In[10]:


y_test_kat.shape[0] == X_test_kat.shape[0]


# In[11]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[12]:


clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=300, alpha=0.0001,
                     solver='sgd', verbose=1, random_state=21, tol=0.000000001)


# In[13]:


clf.fit(X_train_kat, y_train_kat.flatten())


# In[14]:


y_pred_kat = clf.predict(X_test_kat)


# In[15]:


accuracy_score(y_test_kat.flatten(), y_pred_kat)


# ### Test Dataset

# In[16]:


import os
import zipfile
import argparse
import numpy as np
import cPickle as cp
from io import BytesIO
from pandas import Series
import random


# In[17]:


data = np.genfromtxt('testdata.csv', delimiter=',')


# In[18]:


NB_SENSOR_CHANNELS = 10

NORM_MAX_THRESHOLDS = [200, 200, 200, 250000, 250000, 250000, 100, 100, 100, 100]

NORM_MIN_THRESHOLDS = [-200, -200, -200, -250000, -250000, -250000, -100, -100, -100, -100]


# In[19]:


def normalize(data, max_list, min_list):
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


# In[20]:


def process_dataset_file(dataset):
    # Colums are segmentd into features and labels
    data_x = dataset

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

    return data_x


# In[21]:


def generate_data(data):
    data_x = np.empty((0, 10))
    x = process_dataset_file(data)
    data_x = np.vstack((data_x, x))

    return data_x


# In[22]:


test_data = generate_data(data)


# In[23]:


first_conc_x = np.concatenate(test_data[0:15, :])
for i in range(15, test_data.shape[0], 15):
    first_conc_x = np.vstack((first_conc_x, np.concatenate(test_data[i:(i+15), :])))
test_data_kat = first_conc_x


# In[24]:


clf.predict(test_data_kat).flatten()


# In[ ]:





# In[ ]:




