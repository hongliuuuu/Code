from sklearn.externals.six import StringIO
import pydotplus as pydot
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import xlrd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import random
import pylab as pl
import pandas
from sklearn.metrics import accuracy_score

def splitdata(X,Y,ratio,seed):
    '''This function is to split the data into train and test data randomly and preserve the pos/neg ratio'''
    n_samples = X.shape[0]
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    #fint the indices for each class
    indices = []
    for i in classes:
        indice = []
        for j in range(n_samples):
            if y[j] == i:
                indice.append(j)
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
def rf_dis(n_trees, X,Y,train_indices,test_indices,seed):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                 random_state=seed, oob_score=True, n_jobs=-1)
    clf = clf.fit(X[train_indices], Y[train_indices])
    pred = clf.predict(X[test_indices])
    weight = clf.score(X[test_indices], Y[test_indices])
    #print(1 - clf.oob_score_)
    n_samples = X.shape[0]
    dis = np.zeros((n_samples,n_samples))
    for i in range(n_samples):
        dis[i][i] = 0
    res = clf.apply(X)
    for i in range(n_samples):
        for j in range(i+1,n_samples):
            a = np.ravel(res[i])
            b = np.ravel(res[j])
            score = a == b
            d = float(score.sum())/n_trees
            dis[i][j]  =dis[j][i] = d
    X_features1 = np.transpose(dis)
    X_features2 = X_features1[train_indices]
    X_features3 = np.transpose(X_features2)
    return X_features3[train_indices],X_features3[test_indices],weight,pred

def knn(n_neb, train_x, train_y, test_x, test_y):
    clf =KNeighborsClassifier(n_neighbors=n_neb, n_jobs=-1)
    clf.fit(train_x, train_y)
    test_error = clf.score(test_x, test_y)
    test_auc = clf.predict_proba(test_x)
    return test_error

def onn(test_x, train_y, test_y):
    n_s = test_x.shape[0]
    l = []
    for i in range(n_s):
        min = np.min(test_x[i])
        #find the positition of min
        p = test_x[i].index(min)
        l.append(train_y[p])
    s = accuracy_score(test_y, l)
    return s

url = 'totalbbc.csv'
dataframe = pandas.read_csv(url)  # , header=None)
array = dataframe.values
X = array[:, 1:]
# 3 views Y = [1]*414+[2]*307+[3]*380+[4]*351+[5]*376
Y = [1] * 482 + [2] * 359 + [3] * 401 + [4] * 370 + [5] * 400
Y = np.array(Y)
Y = Y.transpose()

# To be corrected
Xnew1 = X[:, 0:6831]
Xnew2 = X[:, 6831:]




err1 = []
err2 = []
tr = []

iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
for  i in range(51):
    ntrees = 20*(i+1)
    seed = 1000+i
    err = []
    errr = []
    for j in range(20):
        se = 1000+j
        train_in, test_in = splitdata(X, Y, 0.7, seed=se)
        X_features_train1, X_features_test1, w1, pred1 = rf_dis(n_trees=ntrees, X=Xnew4, Y=Y, train_indices=train_in,
                                                            test_indices=test_in, seed=se)
        e = knn(n_neb=1,train_x=X_features_train1, train_y=Y[train_in],test_x=X_features_test1,test_y=Y[test_in])

        print(i,se,ntrees)
        e1 = w1
        err.append(e)
        errr.append(e1)
    err1.append(np.mean(err))
    err2.append(np.mean(errr))
    tr.append(ntrees)
print(err1,err2,tr)
pl.plot(tr, err1)# use pylab to plot x and y
pl.plot(tr, err2)# use pylab to plot x and y
pl.legend(['RFDIS', 'RF'], loc='upper left')
pl.show()# show the plot on the screen

'''
clf = RandomForestClassifier(n_estimators=500,
                                 random_state=1000, oob_score=True, n_jobs=-1)
clf.fit(X, Y)
trees = clf.estimators_
ld = [estimator.tree_.max_depth for estimator in clf.estimators_]
print(np.max(ld))
dot_data = StringIO()
tree.export_graphviz(trees[3], out_file=dot_data,

filled=True, rounded=True,proportion =True,
special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('BBC2.pdf')
'''