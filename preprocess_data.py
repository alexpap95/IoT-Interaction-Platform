import numpy as np
import pickle as pickle


# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 13

def process_dataset_file(dataset):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Colums are segmentd into features and labels
    data = np.genfromtxt(dataset, delimiter=',')
    N = 15 # Blocks of N rows
    M,n = data.shape[0]//N, data.shape[1]
    np.random.shuffle(data.reshape(M,-1,n))
    
    data_x = data[:, 1:14]
    data_y = data[:, 0]

    data_y = data_y.astype(int)

#    # Perform linear interpolation
#    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
#
#    # Remaining missing data are converted to zero
#    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
#    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)
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
    count = len(open(data).readlines())
    nb_training_samples = int(round(0.8*count))
    X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]

    print ("Final dataset with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))
    obj = [(X_train, y_train), (X_test, y_test)]
    f = open(target_filename, 'wb')
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


