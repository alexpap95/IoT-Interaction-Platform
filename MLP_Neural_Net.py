import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib



def load_dataset(filename):
    
    data = np.genfromtxt(filename, delimiter=',')
    N = 15 # Blocks of N rows
    M,n = data.shape[0]//N, data.shape[1]
    np.random.shuffle(data.reshape(M,-1,n))
    data_x = data[:, 1:14]
    data_y = data[:, 0]
    data_y = data_y.astype(int)

    # Dataset is segmented into train and test
    count = len(open(filename).readlines())
    nb_training_samples = int(round(0.8*count))
    X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]
    
    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('dataset.csv')


first_conc_x = np.concatenate(X_train[0:15, :])
for i in range(15, X_train.shape[0], 15):
    first_conc_x = np.vstack((first_conc_x, np.concatenate(X_train[i:(i+15), :])))
X_train_mlp = first_conc_x


first_conc_y = np.unique(y_train[0:15])[0]
for i in range(15, y_train.shape[0], 15):
    first_conc_y = np.vstack((first_conc_y, np.unique(y_train[i:(i+15)])[0]))
y_train_mlp = first_conc_y


first_conc_x = np.concatenate(X_test[0:15, :])
for i in range(15, X_test.shape[0], 15):
    first_conc_x = np.vstack((first_conc_x, np.concatenate(X_test[i:(i+15), :])))
X_test_mlp = first_conc_x


first_conc_y = np.unique(y_test[0:15])[0]
for i in range(15, y_test.shape[0], 15):
    first_conc_y = np.vstack((first_conc_y, np.unique(y_test[i:(i+15)])[0]))
y_test_mlp = first_conc_y


clf = MLPClassifier(activation='tanh', hidden_layer_sizes=(100,100,100), max_iter=200, alpha=0.001,
                     solver='adam', verbose=1, random_state=21, tol=0.000000001)


clf.fit(X_train_mlp, y_train_mlp.flatten())

y_pred_mlp = clf.predict(X_test_mlp)
print (classification_report(y_test_mlp.flatten(), y_pred_mlp))
joblib.dump(clf, 'MLP.joblib')







