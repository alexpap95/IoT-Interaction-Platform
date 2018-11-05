import numpy as np
from pandas import Series
from sklearn.externals import joblib

clf = joblib.load('MLP.joblib')

NB_SENSOR_CHANNELS = 10

NORM_MAX_THRESHOLDS = [200, 200, 200, 250000, 250000, 250000, 100, 100, 100, 100]

NORM_MIN_THRESHOLDS = [-200, -200, -200, -250000, -250000, -250000, -100, -100, -100, -100]

t=r=c=0

def normalize(data, max_list, min_list):
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def process(data):
    # Colums are segmentd into features and labels
    data_x = data
    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0
    # All sensor channels are normalized
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)
    return data_x


def generate_data(data):
    global t,r,c
    data_x = process(data)
    data_x = data_x.reshape(1,-1)
    if (int(clf.predict(data_x))==1):
        t+=1
        r=c=0
        if (t==4):
            print ("Left to Right Twist")
            t=0
    elif ((int(clf.predict(data_x))==2)):
        r+=1
        t=c=0
        if (r==4):
            print ("Side Raise")
            r=0
    elif ((int(clf.predict(data_x))==3)):
        c+=1
        t=r=0
        if (c==4):
            print ("Straight to Left Curl")
            c=0
    else:
        t=r=c=0
   
