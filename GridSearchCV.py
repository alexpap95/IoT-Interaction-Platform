# Results of kaggle competition are evaluated using the logloss
# metric, hence this score metric will be used for gridsearchcv 

from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

def svm_superparam_selection(X_train, X_test, y_train, y_test, nfolds, score_evals):
    parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,100,100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.05],
        'learning_rate': ['constant'],
    }
#    param_grid = {
#            'C': [0.01, 0.1, 1],
#            'kernel': ['rbf', 'sigmoid', 'poly'],
#            'gamma': [0.01, 0.1] }
    bestscore_dict = {}
    mlp=MLPClassifier(max_iter=200)
    for score in score_evals:
        grid_search_clf = GridSearchCV(mlp, parameter_space, cv=nfolds, scoring=score, verbose=3, n_jobs=3)
        grid_search_clf.fit(X_train, y_train)
        print('# Tuning hyper-parameters for %s' % score)
        print('Best parameters found based on training set')
        print(grid_search_clf.best_params_)
        bestscore_dict[score] = grid_search_clf.best_params_
        mean_scores = grid_search_clf.cv_results_['mean_test_score']
        std_scores = grid_search_clf.cv_results_['std_test_score']
        print('Grid scores on training set:')
        for mean, std, params in zip(mean_scores, std_scores, grid_search_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        # predict classifier's output
        print('Detailed classification report:')
        print('Scores based on full test set')
        y_pred = grid_search_clf.predict(X_test)
        print(classification_report(y_test, y_pred))
    
    return bestscore_dict


def main():
    
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('dataset.csv')
    # hyperparameter optimization for SVM classifier
    nfolds = 5
    scoring_evaluators = ['accuracy']
    svm_superparam_selection(X_train, X_test, y_train, y_test, nfolds, scoring_evaluators)


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


if __name__=='__main__':
    main()
