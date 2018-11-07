# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:42:01 2018

@author: DELL
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle as pickle
from pandas import Series
import random

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 10


def process_dataset_file(dataset):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Colums are segmentd into features and labels
    data = np.genfromtxt(dataset, delimiter=',')
    blocksize = 15
    blocks = [data[i:i+blocksize] for i in range(0,len(data),blocksize)]
    random.shuffle(blocks)
    data[:] = [b for bs in blocks for b in bs]
    data_x = data[:, 2:12]
    data_y = data[:, 0]

    data_y = data_y.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    return data_x, data_y

def generate_data(data, target_filename):
    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param target_filename: string
        Processed file
    """
    scaler = StandardScaler()
    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))

    x, y = process_dataset_file(data)
    data_x = np.vstack((data_x, x))
    data_y = np.concatenate([data_y, y])

    # Dataset is segmented into train and test
    count = len(open(data).readlines())
    nb_training_samples = int(round(0.8*count))
    X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]
    scale=scaler.fit(X_train)
    print (scale.mean_)
    X_train=scale.transform(X_train)
    X_test=scale.transform(X_test)
    print ("Final dataset with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))
    obj = [(X_train, y_train), (X_test, y_test)]
    f = open(target_filename, 'wb')
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

