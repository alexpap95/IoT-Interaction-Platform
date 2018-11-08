import numpy as np
import pickle as pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from preprocess_data import generate_data
from sklearn.externals import joblib

def load_dataset(filename):

    f = open(filename, 'rb')
    data = pickle.load(f)
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


generate_data('reduced_dataset.csv', 'dataset.data')
print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('dataset.data')


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


clf = MLPClassifier(activation='tanh', hidden_layer_sizes=(100,100,100), max_iter=200, alpha=0.0001,
                     solver='adam', verbose=1, random_state=21, tol=0.00000001)


clf.fit(X_train_mlp, y_train_mlp.flatten())

y_pred_mlp = clf.predict(X_test_mlp)
print (accuracy_score(y_test_mlp.flatten(), y_pred_mlp))
joblib.dump(clf, 'MLP.joblib')







