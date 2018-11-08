import numpy as np
from pandas import Series
from sklearn.externals import joblib

clf = joblib.load('MLP.joblib')
data = np.genfromtxt('testdata.csv', delimiter=',')

NB_SENSOR_CHANNELS = 10

NORM_MAX_THRESHOLDS = [200, 200, 200, 250000, 250000, 250000, 100, 100, 100, 100]

NORM_MIN_THRESHOLDS = [-200, -200, -200, -250000, -250000, -250000, -100, -100, -100, -100]


def normalize(data, max_list, min_list):
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def process_dataset_file(dataset):
    # Colums are segmentd into features and labels
    data_x = dataset[:,2:12]

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    
    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0
    
    # All sensor channels are normalized
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)
    
    return data_x


def generate_data(data):
    data_x = np.empty((0, 10))
    x = process_dataset_file(data)
    data_x = np.vstack((data_x, x))

    return data_x

test_data = generate_data(data)

first_conc_x = np.concatenate(test_data[0:15, :])
for i in range(15, test_data.shape[0], 15):
    first_conc_x = np.vstack((first_conc_x, np.concatenate(test_data[i:(i+15), :])))
test_data_mlp = first_conc_x

print (clf.predict(test_data_mlp).flatten())