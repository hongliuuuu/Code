from sklearn.externals.six import StringIO
import pydotplus as pydot
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import xlrd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import random
import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier

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
def rf_dis(n_trees, X,Y,train_indices,test_indices,seed, mleaf, mf):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                 random_state=seed, oob_score=True, n_jobs=-1,min_samples_leaf = mleaf, max_features =mf )
    clf = clf.fit(X[train_indices], Y[train_indices])
    #pred = clf.predict(X[test_indices])
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
    return X_features3[train_indices],X_features3[test_indices],weight
def EXrf_dis(n_trees, X,Y,train_indices,test_indices,seed, mleaf, mf):
    clf = ExtraTreesClassifier(n_estimators=n_trees,
                                 random_state=seed, oob_score=True, n_jobs=-1,min_samples_leaf = mleaf, max_features =mf )
    clf = clf.fit(X[train_indices], Y[train_indices])
    #pred = clf.predict(X[test_indices])
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
    return X_features3[train_indices],X_features3[test_indices],weight
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
        li = test_x[i]
        min = np.min(li)
        #find the positition of min
        p = li.tolist().index(min)
        l.append(train_y[p])
    s = accuracy_score(test_y, l)
    return s
def RF(n_trees,  seed, train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                  random_state = seed, oob_score=True)
    clf = clf.fit(train_x,train_y)
    oob_error = 1 - clf.oob_score_
    test_error = clf.score(test_x,test_y)
    test_auc = clf.predict_proba(test_x)
    #filename = './tmp1/RF_%d_.pkl'%seed
    #_ = joblib.dump(clf, filename, compress=9)
    return test_error

def ExRF(n_trees,  seed, train_x, train_y, test_x, test_y):
    clf = ExtraTreesClassifier(n_estimators=n_trees,
                                  random_state = seed, oob_score=True)
    clf = clf.fit(train_x,train_y)
    oob_error = 1 - clf.oob_score_
    test_error = clf.score(test_x,test_y)
    test_auc = clf.predict_proba(test_x)
    #filename = './tmp1/RF_%d_.pkl'%seed
    #_ = joblib.dump(clf, filename, compress=9)
    return test_error
'''
X = [[0,0.2,0.6,0.8],
     [0.2,0,0.65,0.7],
     [0.6,0.65,0,0.1],
     [0.8,0.7,0.1,0]]
Y = [0,0,1,1]
test_x = [[0.2,0.1,0.5,0.6],
          [0.6,0.5,0.4,0.1]]
test_x = np.array(test_x)
Y = np.array(Y)
t_Y = Y.transpose()
test_y = [0,1]
test_y = np.array(test_y)
test_y = test_y.transpose()
print(onn(test_x,t_Y,test_y))
'''
url = 'text_pr_1.csv'
dataframe = pandas.read_csv(url, header=None)
array = dataframe.values
X = array
Y = pandas.read_csv('label_progression.csv', header=None)
Y = Y.values
Y = np.ravel(Y)
print(Y.shape)

for i in range(4):
    url = 'text_pr_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X1 = array
    print(X1.shape)
    X = np.concatenate((X, X1), axis=1)

Progression1 = X[:, 0:1680]
Progression2 = X[:, 1680:3360]
Progression3 = X[:, 3360:5040]
Progression4 = X[:, 5040:6720]
Progression5 = X[:, 6720:6745]
ProgressionY = Y
url = 'text_lg_1.csv'
dataframe = pandas.read_csv(url, header=None)
array = dataframe.values
X = array
Y = pandas.read_csv('label_lowGrade.csv', header=None)
Y = Y.values
Y = np.ravel(Y)
print(Y.shape)

for i in range(4):
    url = 'text_lg_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X1 = array
    print(X1.shape)
    X = np.concatenate((X, X1), axis=1)

lowGrade1 = X[:, 0:1680]
lowGrade2 = X[:, 1680:3360]
lowGrade3 = X[:, 3360:5040]
lowGrade4 = X[:, 5040:6720]
lowGrade5 = X[:, 6720:6745]
lowGradeY = Y
url = 'text_nonIDH1_1.csv'
dataframe = pandas.read_csv(url, header=None)
array = dataframe.values
X = array
Y = pandas.read_csv('label_nonIDH1.csv', header=None)
Y = Y.values
Y = np.ravel(Y)
print(Y.shape)

for i in range(4):
    url = 'text_nonIDH1_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X1 = array
    print(X1.shape)
    X = np.concatenate((X, X1), axis=1)

nonIDH1 = X[:, 0:1680]
nonIDH2 = X[:, 1680:3360]
nonIDH3 = X[:, 3360:5040]
nonIDH4 = X[:, 5040:6720]
nonIDH5 = X[:, 6720:6745]
nonIDHY = Y
url = 'text_id_1.csv'
dataframe = pandas.read_csv(url, header=None)
array = dataframe.values
X = array
Y = pandas.read_csv('label_IDHCodel.csv', header=None)
Y = Y.values
Y = np.ravel(Y)
print(Y.shape)

for i in range(4):
    url = 'text_id_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X1 = array
    print(X1.shape)
    X = np.concatenate((X, X1), axis=1)

IDHCodel1 = X[:, 0:1680]
IDHCodel2 = X[:, 1680:3360]
IDHCodel3 = X[:, 3360:5040]
IDHCodel4 = X[:, 5040:6720]
IDHCodel5 = X[:, 6720:6745]
IDHCodelY = Y
url = './MyData.csv'
dataframe = pandas.read_csv(url)#, header=None)
array = dataframe.values
X = array[:,1:]
Y = pandas.read_csv('./MyDatalabel.csv')
Y = Y.values
Y = Y[:,1:]
Y = Y.transpose()
Y = np.ravel(Y)
n_samples = X.shape[0]
n_features = X.shape[1]
for i in range(len(Y)):
    if Y[i] == 4:
        Y[i]=1

Metabo1 = X[:, 0:2]
Metabo2 = X[:, 2:21]
Metabo3 = X[:, 21:]
MetaboY = Y
'''
path = './LSVT.xlsx'
path1 = './LSVTlabel.xlsx'

workbook = xlrd.open_workbook(path)
workbook1 = xlrd.open_workbook(path1)
worksheet = workbook.sheet_by_index(0)
worksheet1 = workbook1.sheet_by_index(0)
# Change this depending on how many header rows are present
# Set to 0 if you want to include the header data.
offset = 0

data = []
for i, row in enumerate(range(worksheet.nrows)):
    if i <= offset:  # (Optionally) skip headers
        continue
    r = []
    for j, col in enumerate(range(worksheet.ncols)):
        r.append(worksheet.cell_value(i, j))
    data.append(r)

label = []
for i, row in enumerate(range(worksheet1.nrows)):
    if i <= offset:  # (Optionally) skip headers
        continue
    r = []
    for j, col in enumerate(range(worksheet1.ncols)):
        r.append(worksheet1.cell_value(i, j))
    label += r
X = np.array(data)
Y = np.array(label)

LSVT1 = X[:,0:51]
LSVT2 =X[:, 51:83]
LSVT3 = X[:, 124:]
LSVT4 = X[:, 83:124]
LSVTY = Y
url = 'Cal7_1.csv'
dataframe = pandas.read_csv(url)  # , header=None)
array = dataframe.values
X = array[:, 1:]

for i in range(5):
    url = 'Cal7_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url)  # , header=None)
    array = dataframe.values
    X1 = array[:, 1:]
    X = np.concatenate((X, X1), axis=1)
Y = pandas.read_csv('Cal7_label.csv')
Y = Y.values

Y = Y[:, 1:]
# Y = Y.transpose()
Y = np.ravel(Y)
Cal71 = X[:, 0:48]
Cal72 = X[:, 48:88]
Cal73 = X[:, 88:342]
Cal74 = X[:, 342:2326]
Cal75 = X[:, 2326:2838]
Cal76 = X[:, 2838:]
Cal7Y = Y
'''
def mcode(ite):
    if ite ==0:
        file = "nonIDH1"
        dataX = nonIDH1
        dataY = nonIDHY
    if ite == 1:
        file = "nonIDH5"
        dataX = nonIDH5
        dataY = nonIDHY
    if ite ==2:
        file = "IDHCodel2"
        dataX = IDHCodel2
        dataY = IDHCodelY
    if ite == 3:
        file = "IDHCodel4"
        dataX = IDHCodel4
        dataY = IDHCodelY
    if ite ==4:
        file = "lowGrade3"
        dataX = lowGrade3
        dataY = lowGradeY
    if ite == 5:
        file = "lowGrade4"
        dataX = lowGrade4
        dataY = lowGradeY
    if ite ==6:
        file = "Progression1"
        dataX = Progression1
        dataY = ProgressionY
    if ite == 7:
        file = "Progression5"
        dataX = Progression5
        dataY = ProgressionY
    if ite ==8:
        file = "Metabo2"
        dataX = Metabo2
        dataY = MetaboY
    if ite == 9:
        file = "Metabo3"
        dataX = Metabo3
        dataY = MetaboY
    perfonn = [[[0 for k in range(4)] for j in range(5)] for i in range(2)]
    perfrf = [[[0 for k in range(4)] for j in range(5)] for i in range(2)]
    perfrfdis = [[[0 for k in range(4)] for j in range(5)] for i in range(2)]

    for i in range(2):
        if i == 0:
            mtry = 1
        else:
            mtry = "auto"
        for  j in range(5):
            if j == 0:
                ntrees = 50
            if j == 1:
                ntrees = 100
            if j == 2:
                ntrees = 500
            if j == 3:
                ntrees = 800
            if j == 4:
                ntrees = 1000
            for k in range(4):
                if k == 0:
                    minleaf = 0.5
                if k == 1:
                    minleaf = 0.3
                if k == 2:
                    minleaf = 0.1
                if k == 3:
                    minleaf = 1

                err = []
                errr = []
                eknn = []
                erf = []
                se = 1000
                for ite in range(10):
                    se = se+1
                    train_in, test_in = splitdata(dataX,dataY,0.5,se)

                    X_features_train1, X_features_test1, w1 = rf_dis(n_trees=ntrees, X=dataX, Y=dataY, train_indices=train_in,
                                                                        test_indices=test_in, seed=se,mleaf = minleaf,mf = mtry)
                    e = onn( train_y=dataY[train_in],test_x=X_features_test1,test_y=dataY[test_in])
                    #e2 = knn(n_neb=1, train_x=X_features_train1, train_y=Y[train_in], test_x=X_features_test1, test_y=Y[test_in])
                    e3 = RF(n_trees=500,seed=se,train_x=X_features_train1, train_y=dataY[train_in], test_x=X_features_test1, test_y=dataY[test_in])
                    #print(i,se,ntrees)
                    e1 = w1
                    err.append(e)
                    errr.append(e1)
                    #eknn.append(e2)
                    erf.append(e3)
                #print(len(err),len(errr),len(eknn),len(erf))
                #err1.append(np.mean(err)) #performance of onn
                perfonn[i][j][k]= np.mean(err)
                #err2.append(np.mean(errr))#performance of RF
                perfrf[i][j][k] = np.mean(errr)
                #err3.append(np.mean(eknn))
                #err4.append(np.mean(erf))#performance of RF dis
                perfrfdis[i][j][k] = np.mean(erf)


    #prediction = pandas.DataFrame(data).to_csv(file)
    filename1 = file + "onnRF"
    filename2 = file + "rfRF"
    filename3 = file + "rfdisRF"
    np.save(filename1, perfonn)
    np.save(filename2, perfrf)
    np.save(filename3, perfrfdis)

if __name__ == '__main__':
    Parallel(n_jobs=10)(delayed(mcode)(ite=i) for i in range(10))
'''
pl.xlabel('tree number')

pl.plot(tr, err1)# use pylab to plot x and y
pl.plot(tr, err2)# use pylab to plot x and y
pl.plot(tr, err3)# use pylab to plot x and y
pl.plot(tr, err4)# use pylab to plot x and y
pl.legend(['1nn', 'RF', "Dis1NN"], loc='upper left')
pl.show()# show the plot on the screen'''


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