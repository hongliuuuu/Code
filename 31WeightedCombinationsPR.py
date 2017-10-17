from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel,laplacian_kernel,chi2_kernel,linear_kernel,polynomial_kernel,cosine_similarity
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xlrd
import xlrd
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
import re
from math import floor
from joblib import Parallel, delayed

def floored_percentage(val, digits):
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}\%\pm '.format(digits, floor(val) / 10 ** digits)
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
        #print(len(indice))
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
    clf = RandomForestClassifier(n_estimators=500,
                                 random_state=seed, oob_score=True, n_jobs=1)
    clf = clf.fit(X[train_indices], Y[train_indices])
    pred = clf.predict(X[test_indices])
    prob = clf.predict_proba(X[test_indices])
    weight = clf.score(X[test_indices], Y[test_indices])
    #print(1 - clf.oob_score_)
    n_samples = X.shape[0]
    dis = np.zeros((n_samples,n_samples))
    for i in range(n_samples):
        dis[i][i] = 1
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
    return X_features3[train_indices],X_features3[test_indices],weight,pred,prob

def gama_patatune(train_x,train_y,c):
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [0.0625, 0.125,0.25, 0.5, 1, 2, 5 ,7, 10, 12 ,15 ,17 ,20] }]
    clf = GridSearchCV(SVC(C=c), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    return clf.best_params_['gamma']


def relf(n_neb, n_feat, trainx, trainy,testx):
    fs = ReliefF(n_features_to_select=n_feat, n_neighbors=n_neb,discrete_threshold=10, n_jobs=1)
    fs.fit(trainx, trainy)
    ind = fs.transform(trainx)
    return ind

def lsvm_rfe(c,n_feat,trainX,trainy, testX):
    svc = SVC(kernel="linear", C=c)
    rfe = RFE(estimator=svc, n_features_to_select=n_feat, step=1)
    rfe.fit(trainX, trainy)
    train_X = rfe.transform(trainX)
    test_X = rfe.transform(testX)
    return train_X,test_X
def RF(n_trees,  seed, train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                  random_state = seed, oob_score=True)
    clf = clf.fit(train_x,train_y)
    oob_error = 1 - clf.oob_score_
    test_error = clf.score(test_x,test_y)
    test_auc = clf.predict_proba(test_x)
    #filename = './tmp1/RF_%d_.pkl'%seed
    #_ = joblib.dump(clf, filename, compress=9)
    return test_error, test_auc
def selected_f(n_features):
    if n_features>1000:
        n = 25
    elif n_features>100:
        n = int(n_features*0.03)
    elif n_features >75:
        n = int(n_features * 0.1)
    else :
        n = int(n_features * 0.4)
    return n
def nLsvm_patatune(train_x,train_y,test_x, test_y):
    tuned_parameters = [
        {'kernel': ['precomputed'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    print(clf.score(test_x,test_y))
    return clf.best_params_['C']
def Lsvm_patatune(train_x,train_y):
    tuned_parameters = [
        {'kernel': ['linear'], 'C': [0.01,0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1, probability=True), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    return clf.best_params_['C']

def weightedComb(Y, W):
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    # fint the indices for each class
    indices = []

    for i in classes:
        pro = 0
        indice = []
        for j in range(len(y)):
            if y[j] == i:
                indice.append(j)
                pro = pro + W[j]
        indices.append(pro)
    ind = (list(indices)).index(max(indices))
    return classes[ind]

def mcode(ite):
    R = 0.5
    testfile = open("31WeightedSYS800IDH1%f_%f.txt" % (R,ite), 'w')
    numberofclass = 2

    e8 = []

    erfsvm = []
    elaterf = []
    elaterfdis = []


    seed = 1000 + ite
    #data reading
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

    Xnew1 = X[:, 0:1680]
    Xnew2 = X[:, 1680:3360]
    Xnew3 = X[:, 3360:5040]
    Xnew4 = X[:, 5040:6720]
    Xnew5 = X[:, 6720:6745]
    #Xnew6 = X[:, 6745:]
    #combine 2 views together
    Xnew6 = np.concatenate((Xnew1, Xnew2), axis=1)
    Xnew7 = np.concatenate((Xnew1, Xnew3), axis=1)
    Xnew8 = np.concatenate((Xnew1, Xnew4), axis=1)
    Xnew9 = np.concatenate((Xnew1, Xnew5), axis=1)

    Xnew10 = np.concatenate((Xnew2, Xnew3), axis=1)
    Xnew11 = np.concatenate((Xnew2, Xnew4), axis=1)
    Xnew12 = np.concatenate((Xnew2, Xnew5), axis=1)

    Xnew13 = np.concatenate((Xnew3, Xnew4), axis=1)
    Xnew14 = np.concatenate((Xnew3, Xnew5), axis=1)

    Xnew15 = np.concatenate((Xnew4, Xnew5), axis=1)

    # combine 3 views together
    Xnew16 = np.concatenate((Xnew1, Xnew2, Xnew3), axis=1)
    Xnew17 = np.concatenate((Xnew1, Xnew2, Xnew4), axis=1)
    Xnew18 = np.concatenate((Xnew1, Xnew2, Xnew5), axis=1)
    Xnew19 = np.concatenate((Xnew1, Xnew3, Xnew4), axis=1)
    Xnew20 = np.concatenate((Xnew1, Xnew3, Xnew5), axis=1)
    Xnew21 = np.concatenate((Xnew1, Xnew4, Xnew5), axis=1)

    Xnew22 = np.concatenate((Xnew2, Xnew3, Xnew4), axis=1)
    Xnew23 = np.concatenate((Xnew2, Xnew3, Xnew5), axis=1)
    Xnew24 = np.concatenate((Xnew2, Xnew4, Xnew5), axis=1)

    Xnew25 = np.concatenate((Xnew3, Xnew4, Xnew5), axis=1)

    # combine 4 views together
    Xnew26 = np.concatenate((Xnew1, Xnew2, Xnew3, Xnew4), axis=1)
    Xnew27 = np.concatenate((Xnew1, Xnew2, Xnew3, Xnew5), axis=1)
    Xnew28 = np.concatenate((Xnew1, Xnew2, Xnew5, Xnew4), axis=1)
    Xnew29 = np.concatenate((Xnew1, Xnew3, Xnew5, Xnew4), axis=1)
    Xnew30 = np.concatenate((Xnew2, Xnew3, Xnew5, Xnew4), axis=1)

    # combine 5 views together
    Xnew31 = np.concatenate((Xnew1, Xnew2, Xnew3, Xnew4, Xnew5), axis=1)

    print("datasize")
    print(X.shape)
    train_indices, test_indices = splitdata(X=X, Y=Y, ratio=R, seed=seed)

    X_features_train = []
    X_features_test = []
    pred = []
    pre = []
    prorf = []
    prodis = []
    eprf = []
    eprfdis = []
    weight = []


    for ii in range(31):
        if ii==0:
            Xdata = Xnew1
        elif ii == 1:
            Xdata = Xnew2
        elif ii == 2:
            Xdata = Xnew3
        elif ii == 3:
            Xdata = Xnew4
        elif ii == 4:
            Xdata = Xnew5
        elif ii == 5:
            Xdata = Xnew6
        elif ii == 6:
            Xdata = Xnew7
        elif ii == 7:
            Xdata = Xnew8
        elif ii == 8:
            Xdata = Xnew9
        elif ii == 9:
            Xdata = Xnew10
        elif ii==10:
            Xdata = Xnew11
        elif ii == 11:
            Xdata = Xnew12
        elif ii == 12:
            Xdata = Xnew13
        elif ii == 13:
            Xdata = Xnew14
        elif ii == 14:
            Xdata = Xnew15
        elif ii == 15:
            Xdata = Xnew16
        elif ii == 16:
            Xdata = Xnew17
        elif ii == 17:
            Xdata = Xnew18
        elif ii == 18:
            Xdata = Xnew19
        elif ii == 19:
            Xdata = Xnew20
        elif ii==20:
            Xdata = Xnew21
        elif ii == 21:
            Xdata = Xnew22
        elif ii == 22:
            Xdata = Xnew23
        elif ii == 23:
            Xdata = Xnew24
        elif ii == 24:
            Xdata = Xnew25
        elif ii == 25:
            Xdata = Xnew26
        elif ii == 26:
            Xdata = Xnew27
        elif ii == 27:
            Xdata = Xnew28
        elif ii == 28:
            Xdata = Xnew29
        elif ii == 29:
            Xdata = Xnew30
        else :
            Xdata = Xnew31

        X_features_train1, X_features_test1, w1, pred1, prob1 = rf_dis(n_trees=500, X=Xdata, Y=Y, train_indices=train_indices,
                                                                test_indices=test_indices, seed=seed)
        w = max(w1-(1/numberofclass),0)
        weight.append(w)
        m12 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_train1, Y[train_indices])
        pre1 = m12.predict(X_features_test1)
        prob2 = m12.predict_proba(X_features_test1)
        X_features_train.append(X_features_train1)
        X_features_test.append(X_features_test1)
        pred.append(pred1)
        pre.append(pre1)
        prorf.append(prob1*w)
        prodis.append(prob2*w)


    """
    #view1

    X_features_train1, X_features_test1,w1,pred1= rf_dis(n_trees=500,  X=Xnew1,Y=Y,  train_indices=train_indices,test_indices=test_indices,seed=seed)
    m12 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train1, Y[train_indices])
    pre1 = m12.predict(X_features_test1)

    #view 2

    X_features_train2, X_features_test2, w2,pred2 = rf_dis(n_trees=500, X=Xnew2,Y=Y, train_indices=train_indices,
                                                           test_indices=test_indices, seed=seed)
    m22 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train2, Y[train_indices])
    pre2 = m22.predict(X_features_test2)


    #view 3

    X_features_train3, X_features_test3, w3,pred3 = rf_dis(n_trees=500, X=Xnew3,Y=Y, train_indices=train_indices,
                                                           test_indices=test_indices, seed=seed)
    m32 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train3, Y[train_indices])
    pre3 = m32.predict(X_features_test3)


    #view 4

    X_features_train4, X_features_test4, w4,pred4 = rf_dis(n_trees=500, X=Xnew4,Y=Y, train_indices=train_indices,
                                                           test_indices=test_indices, seed=seed)
    m42 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train4, Y[train_indices])
    pre4 = m42.predict(X_features_test4)


    # view 5

    X_features_train5, X_features_test5, w5, pred5 = rf_dis(n_trees=500, X=Xnew5, Y=Y, train_indices=train_indices,
                                                            test_indices=test_indices, seed=seed)
    m52 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train5, Y[train_indices])
    pre5 = m52.predict(X_features_test5)

    """









    """
    # Late RF
    resall1 = np.column_stack((pred[0], pred[1], pred[2], pred[3],pred[4],pred[5], pred[6], pred[7], pred[8],pred[9],
    pred[10], pred[11], pred[12], pred[13], pred[14], pred[15], pred[16], pred[17], pred[18], pred[19],
    pred[20], pred[21], pred[22], pred[23], pred[24], pred[25], pred[26], pred[27], pred[28], pred[29], pred[30])) #,pred[5], pred[6], pred[7], pred[8],pred[9],
    #pred[10], pred[11], pred[12], pred[13], pred[14], pred[15], pred[16], pred[17], pred[18], pred[19],
    #pred[20], pred[21], pred[22], pred[23], pred[24], pred[25], pred[26], pred[27], pred[28], pred[29], pred[30]
    Laterf = list(range(len(test_indices)))
    for i in range(len(test_indices)):
        Laterf[i], empty = Counter(resall1[i]).most_common()[0]
    LRF = accuracy_score(Y[test_indices], Laterf)
    elaterf.append(LRF)

    # Late RF dis
    resall = np.column_stack((pre[0], pre[1], pre[2], pre[3],pre[4],pre[5], pre[6], pre[7], pre[8],pre[9],
    pre[10], pre[11], pre[12], pre[13], pre[14], pre[15], pre[16], pre[17], pre[18], pre[19],
    pre[20], pre[21], pre[22], pre[23], pre[24], pre[25], pre[26], pre[27], pre[28], pre[29], pre[30])) #,pre[5], pre[6], pre[7], pre[8],pre[9],
    #pre[10], pre[11], pre[12], pre[13], pre[14], pre[15], pre[16], pre[17], pre[18], pre[19],
    #pre[20], pre[21], pre[22], pre[23], pre[24], pre[25], pre[26], pre[27], pre[28], pre[29], pre[30]
    LSVTres = list(range(len(test_indices)))
    for i in range(len(test_indices)):
        LSVTres[i], empty = Counter(resall[i]).most_common()[0]
    LSVTscore = accuracy_score(Y[test_indices], LSVTres)
    elaterfdis.append(LSVTscore)
    """
    #weighted LRF
    resall1 = np.column_stack((pred[0], pred[1], pred[2], pred[3], pred[4],pred[5], pred[6], pred[7], pred[8],pred[9],
    pred[10], pred[11], pred[12], pred[13], pred[14], pred[15], pred[16], pred[17], pred[18], pred[19],
    pred[20], pred[21], pred[22], pred[23], pred[24], pred[25], pred[26], pred[27], pred[28], pred[29], pred[30]))

    Laterf = list(range(len(test_indices)))
    for i in range(len(test_indices)):
        Laterf[i] = weightedComb(resall1[i],weight)
    LRF = accuracy_score(Y[test_indices], Laterf)
    elaterf.append(LRF)

    #weighted LRFDIS
    resall = np.column_stack((pre[0], pre[1], pre[2], pre[3], pre[4],pre[5], pre[6], pre[7], pre[8],pre[9],
     pre[10], pre[11], pre[12], pre[13], pre[14], pre[15], pre[16], pre[17], pre[18], pre[19],
    pre[20], pre[21], pre[22], pre[23], pre[24], pre[25], pre[26], pre[27], pre[28], pre[29], pre[30]))  # ,pre[5], pre[6], pre[7], pre[8],pre[9],
    # pre[10], pre[11], pre[12], pre[13], pre[14], pre[15], pre[16], pre[17], pre[18], pre[19],
    # pre[20], pre[21], pre[22], pre[23], pre[24], pre[25], pre[26], pre[27], pre[28], pre[29], pre[30]
    LSVTres = list(range(len(test_indices)))
    for i in range(len(test_indices)):
        LSVTres[i] = weightedComb(resall[i],weight)
    LSVTscore = accuracy_score(Y[test_indices], LSVTres)
    elaterfdis.append(LSVTscore)



    #Late RF prob
    mprf = 0
    for le in range(len(prorf)):
        mprf = mprf+prorf[le]
    prfres = []
    for i in range(mprf.shape[0]):
        ind = (list(mprf[i])).index(max(mprf[i]))
        prfres.append(ind)
    score = accuracy_score(Y[test_indices], prfres)
    eprf.append(score)


    # Late RFDIS prob
    mprf = 0
    for le in range(len(prodis)):
        mprf = mprf + prodis[le]
    prfres = []
    for i in range(mprf.shape[0]):
        ind = (list(mprf[i])).index(max(mprf[i]))
        prfres.append(ind)
    score = accuracy_score(Y[test_indices], prfres)
    eprfdis.append(score)

    """
    #multi view
    X_features_trainm = (
                            X_features_train1 + X_features_train2 + X_features_train3 + X_features_train4+X_features_train5 ) / 5
    X_features_testm = (
                           X_features_test1 + X_features_test2 + X_features_test3 + X_features_test4 +X_features_test5) / 5
    """
    X_features_trainm = 0
    X_features_testm = 0
    mm = len(X_features_train)
    for ss in range(mm):
        X_features_trainm = X_features_trainm+X_features_train[ss]*weight[ss]
        X_features_testm = X_features_testm + X_features_test[ss]*weight[ss]
    X_features_trainm = X_features_trainm/mm
    X_features_testm = X_features_testm/mm

    #RFDIS
    mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_trainm, Y[train_indices])
    e8.append(mv.score(X_features_testm, Y[test_indices]))

    #RFSVM
    c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                       test_y=Y[test_indices])
    clf = SVC(C=c, kernel='precomputed')
    clf.fit(X_features_trainm, Y[train_indices])
    erfsvm.append(clf.score(X_features_testm, Y[test_indices]))

    testfile.write("RFSVM&%s pm%s & " % (floored_percentage(np.mean(erfsvm),2), floored_percentage(np.std(erfsvm),2)) + '\n')
    testfile.write("RFDIS &%s pm%s & " % (floored_percentage(np.mean(e8),2), floored_percentage(np.std(e8),2)) + '\n')
    testfile.write(" LATERF&%s pm%s &" % (floored_percentage(np.mean(elaterf),2), floored_percentage(np.std(elaterf),2)) + '\n')
    testfile.write(" LATERFDIS&%s pm%s & " % (floored_percentage(np.mean(elaterfdis),2), floored_percentage(np.std(elaterfdis),2)) + '\n')
    testfile.write(
        " LATEpro&%s pm%s &" % (floored_percentage(np.mean(eprf), 2), floored_percentage(np.std(eprf), 2)) + '\n')
    testfile.write(" LATEprodis&%s pm%s & " % (
    floored_percentage(np.mean(eprfdis), 2), floored_percentage(np.std(eprfdis), 2)) + '\n')

    testfile.close()

if __name__ == '__main__':
    Parallel(n_jobs=4)(delayed(mcode)(ite=i) for i in range(10))