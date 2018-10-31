import os
import zipfile
import argparse
import numpy as np
import cPickle as cp
from io import BytesIO
from pandas import Series
import random

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 10

# Hardcoded thresholds to define global maximums and minimums for every one of the 113 sensor channels employed in the
# OPPORTUNITY challenge
NORM_MAX_THRESHOLDS = [200, 200, 200, 250000, 250000, 250000, 100, 100, 100, 100]

NORM_MIN_THRESHOLDS = [-200, -200, -200, -250000, -250000, -250000, -100, -100, -100, -100]

def normalize(data, max_list, min_list):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param max_list: numpy integer array
        Array containing maximums values for every one of the 113 sensor channels
    :param min_list: numpy integer array
        Array containing minimum values for every one of the 113 sensor channels
    :return:
        Normalized sensor data
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    #     Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data

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

    # All sensor channels are normalized
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)
    return data_x, data_y

def generate_data(data, target_filename):
    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param target_filename: string
        Processed file
    """
    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))

    x, y = process_dataset_file(data)
    data_x = np.vstack((data_x, x))
    data_y = np.concatenate([data_y, y])

    # Dataset is segmented into train and test
    nb_training_samples = 46800
    X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]

    print "Final dataset with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape)
    obj = [(X_train, y_train), (X_test, y_test)]
    f = file(target_filename, 'wb')
    cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
    f.close()

def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description='Preprocess dataset')
    # Add arguments
    parser.add_argument(
        '-i', '--input', type=str, help='Dataset file', required=True)
    parser.add_argument(
        '-o', '--output', type=str, help='Processed data file', required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    dataset = args.input
    target_filename = args.output
    # Return all variable values
    return dataset, target_filename

if __name__ == '__main__':
    data, output= get_args();
    generate_data(data, output)