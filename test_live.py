import numpy as np
from sklearn.externals import joblib

clf = joblib.load('MLP.joblib')

NB_SENSOR_CHANNELS = 13
predictions=np.full(15,4)


def generate_data(data):
    global predictions
    data_x = data.reshape(1,-1)
    predictions = np.delete(predictions, 0)
    predictions = np.append(predictions, int(clf.predict(data_x)))
    counts = np.bincount(predictions)
    if (counts[1]>3):
        print ("Left to Right Twist")
        predictions=np.full(15,4)
    if (counts[2]>3):
        print ("Side Raise")
        predictions=np.full(15,4)
    if (counts[3]>3):
        print ("Straight to Left Curl")
        predictions=np.full(15,4)
            
   
