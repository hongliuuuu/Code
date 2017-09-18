import scipy.io
import numpy as np
import pandas
import random
import time
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from collections import Counter


def splitdata(X,Y,ratio,seed):
    '''This function is to split the data into train and test data randomly and preserve the pos/neg ratio'''
    n_samples = X.shape[0]
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    #fint the indices for each class
    indices = []
    print()
    for i in classes:
        indice = []
        for j in range(n_samples):
            if y[j] == i:
                indice.append(j)
        print(len(indice))
        indices.append(indice)
    train_indices = []
    for i in indices:
        k = int(len(i)*ratio)
        train_indices += (random.Random(seed).sample(i,k=k))
    #find the unused indices
    s = np.bincount(train_indices,minlength=n_samples)
    mask = s==0
    test_indices = np.arange(n_samples)[mask]
    return train_indices,test_indices

def RF(n_trees,  seed, train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                  random_state = seed, oob_score=True,n_jobs=-1)
    clf = clf.fit(train_x,train_y)
    oob_error = 1 - clf.oob_score_
    test_error = clf.score(test_x,test_y)
    test_auc = clf.predict_proba(test_x)
    #filename = './tmp1/RF_%d_.pkl'%seed
    #_ = joblib.dump(clf, filename, compress=9)
    return test_error, test_auc

def testdata(X,Y):
    train, test = splitdata(X, Y, 0.5, 1000)
    train_X = X[train]
    test_X = X[test]
    train_Y = Y[train]
    test_Y = Y[test]

    err, sc = RF(500, 1001, train_X, train_Y, test_X, test_Y)
    print(err)


url = 'text_nonIDH1_1.csv'
dataframe = pandas.read_csv(url , header=None)
array = dataframe.values
X = array
print(X.shape)
Y = pandas.read_csv('label_nonIDH1.csv', header=None)
Y = Y.values
Y = np.ravel(Y)
print(Y.shape)
testdata(X,Y)

for i in range(5):
    url = 'text_nonIDH1_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X1 = array
    print(X1.shape)
    testdata(X1, Y)
    X = np.concatenate((X, X1), axis=1)


#train_indices, test_indices = splitdata(X=X, Y=Y, ratio=0.7, seed=1000 )
n_features = X.shape[1]
n_trees = 500

"""
for i in range(n_features):
    s = X[:,i]
    mn = np.max(s)-np.min(s)
    if mn == 0:
        print(i)
"""
#ind = [138, 302, 306, 632, 633, 1342, 2241, 2338, 3154, 3155, 3383, 4988, 4989, 5345, 5513, 5926, 6016, 6141]
#X = np.delete(X,ind,axis=1)
print(X.shape)
print("hello")
li = []
for i in range(X.shape[1]):
    s = X[:,i]
    mn = np.max(s)-np.min(s)
    if mn == 0:
        li.append(i)
print(li)
#prediction = pandas.DataFrame(X).to_csv('totalflower.csv')

testdata(X,Y)

Xnew1 = X[:, 0:1680]
Xnew2 = X[:, 1680:3360]
Xnew3 = X[:, 3360:5040]
Xnew4 = X[:, 5040:6720]
Xnew5 = X[:, 6720:6745]
Xnew6 = X[:, 6745:]