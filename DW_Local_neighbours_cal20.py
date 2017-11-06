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
    return train_indices,test_indices, len(classes)

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
    return X_features3[train_indices],X_features3[test_indices],weight,pred,prob,clf

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
    testfile = open("SWLN10OnonIDH1%f_%f.txt" % (R,ite), 'w')

    n_neighbours = 10
    n_views = 6
    e8 = []

    erfsvm = []
    elaterf = []
    elaterfdis = []


    seed = 1000 + ite
    #data reading
    url = 'Cal20_1.csv'
    dataframe = pandas.read_csv(url)  # , header=None)
    array = dataframe.values
    X = array[:, 1:]

    for i in range(5):
        url = 'Cal20_' + str(i + 2) + '.csv'
        dataframe = pandas.read_csv(url)  # , header=None)
        array = dataframe.values
        X1 = array[:, 1:]
        X = np.concatenate((X, X1), axis=1)
    Y = pandas.read_csv('Cal20_label.csv')
    Y = Y.values

    Y = Y[:, 1:]
    # Y = Y.transpose()
    Y = np.ravel(Y)

    Xnew1 = X[:, 0:48]
    Xnew2 = X[:, 48:88]
    Xnew3 = X[:, 88:342]
    Xnew4 = X[:, 342:2326]
    Xnew5 = X[:, 2326:2838]
    Xnew6 = X[:, 2838:]
    print("datasize")
    print(X.shape)
    train_indices, test_indices, numberofclass = splitdata(X=X, Y=Y, ratio=R, seed=seed)

    X_features_train = []
    X_features_test = []
    pred = []
    pre = []
    prorf = []
    prodis = []
    eprf = []
    eprfdis = []
    weight = []


    X_features_train1, X_features_test1, w1, pred1, prob1, RFV1 = rf_dis(n_trees=500, X=Xnew1, Y=Y, train_indices=train_indices,
                                                                   test_indices=test_indices, seed=seed)
    X_features_train2, X_features_test2, w2, pred2, prob2, RFV2 = rf_dis(n_trees=500, X=Xnew2, Y=Y,
                                                                         train_indices=train_indices,
                                                                         test_indices=test_indices, seed=seed)
    X_features_train3, X_features_test3, w3, pred3, prob3, RFV3 = rf_dis(n_trees=500, X=Xnew3, Y=Y,
                                                                         train_indices=train_indices,
                                                                         test_indices=test_indices, seed=seed)
    X_features_train4, X_features_test4, w4, pred4, prob4, RFV4 = rf_dis(n_trees=500, X=Xnew4, Y=Y,
                                                                         train_indices=train_indices,
                                                                         test_indices=test_indices, seed=seed)
    X_features_train5, X_features_test5, w5, pred5, prob5, RFV5 = rf_dis(n_trees=500, X=Xnew5, Y=Y,
                                                                         train_indices=train_indices,
                                                                         test_indices=test_indices, seed=seed)
    X_features_train6, X_features_test6, w6, pred6, prob6, RFV6 = rf_dis(n_trees=500, X=Xnew6, Y=Y,
                                                                         train_indices=train_indices,
                                                                         test_indices=test_indices, seed=seed)
    W= [w1,w2,w3,w4,w5,w6]
    weight = [max(w1-(1/numberofclass),0),max(w2-(1/numberofclass),0),max(w3-(1/numberofclass),0),max(w4-(1/numberofclass),0),max(w5-(1/numberofclass),0),max(w6-(1/numberofclass),0)]
    dwRF = np.zeros((len(test_indices), n_views))
    olRF = np.zeros((len(test_indices), n_views))
    oalRF = np.zeros((len(test_indices), n_views))
    oallRF = np.zeros((len(test_indices), n_views))
    prorf = [prob1*weight[0],prob2*weight[1],prob3*weight[2],prob4*weight[3],prob5*weight[4],prob6*weight[5]]
    pro = [prob1,prob2,prob3,prob4,prob5,prob6]
    onelocalRF = np.zeros((len(test_indices), len(prorf)))
    ollRF = np.zeros((len(test_indices), len(prorf)))
    totalmtp = np.zeros((len(test_indices), len(prorf)))
    sumw = np.zeros((len(test_indices), len(prorf)))
    sumW = np.zeros((len(test_indices), len(prorf)))
    for le in range(len(prorf)):
        mprf = prorf[le]
        for i in range(mprf.shape[0]):
            onelocalRF[i, le] = max(mprf[i])
    simplepro = np.zeros((len(test_indices), len(pro)))
    for le in range(len(pro)):
        mprf = pro[le]
        for i in range(mprf.shape[0]):
            simplepro[i, le] = max(mprf[i])
    for nv in range(n_views):
        if nv == 0:
            XFT = X_features_test1
            RFV = RFV1
            XX = Xnew1
        elif nv == 1:
            XFT = X_features_test2
            RFV = RFV2
            XX = Xnew2
        elif nv == 2:
            XFT = X_features_test3
            RFV = RFV3
            XX = Xnew3
        elif nv == 3:
            XFT = X_features_test4
            RFV = RFV4
            XX = Xnew4
        elif nv == 4:
            XFT = X_features_test5
            RFV = RFV5
            XX = Xnew5
        elif nv == 5:
            XFT = X_features_test6
            RFV = RFV6
            XX = Xnew6
        for ind in range(len(test_indices)):
            nei = XFT[ind].argsort()[:n_neighbours]
            newX = XX[nei]
            newY = Y[nei]
            la = RFV.score(newX,newY)
            dwRF[ind,nv] = la
            olRF[ind, nv] = la*weight[nv]
            ollRF[ind, nv] = la * W[nv]
            oalRF[ind, nv] = la + weight[nv]
            oallRF[ind, nv] = la + W[nv]
            totalmtp[ind, nv] = simplepro[ind, nv]*la*W[nv]
            sumw[ind, nv] = simplepro[ind, nv]+la+weight[nv]
            sumW[ind, nv] = simplepro[ind, nv] + la + W[nv]

    #weighted LRF
    resall1 = np.column_stack((pred1, pred2, pred3, pred4, pred5,pred6))
    #resall11 = np.column_stack((prob1, prob2, prob3, prob4, prob5))
    Laterf = list(range(len(test_indices)))
    Lateswrf = list(range(len(test_indices)))
    Latedwrf = list(range(len(test_indices)))
    Latedwrfol = list(range(len(test_indices)))
    Latedwrfoal = list(range(len(test_indices)))
    Latedwrfoall = list(range(len(test_indices)))
    Lateolrf = list(range(len(test_indices)))
    Lollrf = list(range(len(test_indices)))
    Ltotalrf = list(range(len(test_indices)))
    Lsumwrf = list(range(len(test_indices)))
    LsumWrf = list(range(len(test_indices)))
    ww = [1, 1, 1, 1, 1,1]
    for i in range(len(test_indices)):

        Laterf[i] = weightedComb(resall1[i],ww)
        Lateswrf[i] = weightedComb(resall1[i],W)
        Latedwrf[i] = weightedComb(resall1[i],dwRF[i])
        Latedwrfol[i] = weightedComb(resall1[i], olRF[i])
        Latedwrfoal[i] = weightedComb(resall1[i], oalRF[i])
        Lateolrf[i] = weightedComb(resall1[i], onelocalRF[i])
        Latedwrfoall[i]  = weightedComb(resall1[i], oallRF[i])
        Lollrf[i] = weightedComb(resall1[i], ollRF[i])
        Ltotalrf[i] = weightedComb(resall1[i], totalmtp[i])
        Lsumwrf[i] = weightedComb(resall1[i], sumw[i])
        LsumWrf[i] = weightedComb(resall1[i], sumW[i])
        #print(ww)
        #print(Laterf[i])
        #print(W)
        #print(Lateswrf[i])
        #print(dwRF[i])
        #print(Latedwrf[i])
        #print(olRF[i])
        #print(Latedwrfol[i])
        #print(oalRF[i])
        #print(Latedwrfoal[i])

        '''yt = Y[test_indices]
        if yt[i]!=Laterf[i]:
            print(resall1[i])
            print(resall11[i])
            print(dwRF[i])
            print(Laterf[i])'''

    LRF = accuracy_score(Y[test_indices], Laterf)
    elaterf.append(LRF)
    LSW = accuracy_score(Y[test_indices], Lateswrf)
    LDW = accuracy_score(Y[test_indices], Latedwrf)
    LOL = accuracy_score(Y[test_indices], Latedwrfol)
    LAL = accuracy_score(Y[test_indices], Latedwrfoal)
    Lone = accuracy_score(Y[test_indices], Lateolrf)
    OALL =  accuracy_score(Y[test_indices], Latedwrfoall)
    OLL = accuracy_score(Y[test_indices], Lollrf)
    Tota = accuracy_score(Y[test_indices], Ltotalrf)
    Sumw = accuracy_score(Y[test_indices], Lsumwrf)
    SumW = accuracy_score(Y[test_indices], LsumWrf)







    #testfile.write("RFSVM&%s pm%s & " % (floored_percentage(np.mean(erfsvm),2), floored_percentage(np.std(erfsvm),2)) + '\n')
    #testfile.write("RFDIS &%s pm%s & " % (floored_percentage(np.mean(e8),2), floored_percentage(np.std(e8),2)) + '\n')
    testfile.write(" LRF&%s pm%s &" % (floored_percentage(np.mean(elaterf),2), floored_percentage(np.std(elaterf),2)) + '\n')
    #testfile.write(" LATERFDIS&%s pm%s & " % (floored_percentage(np.mean(elaterfdis),2), floored_percentage(np.std(elaterfdis),2)) + '\n')
    testfile.write(" LSW&%s pm%s &" % (floored_percentage(np.mean(LSW),2), floored_percentage(np.std(LSW),2)) + '\n')
    testfile.write(" LDW&%s pm%s &" % (floored_percentage(np.mean(LDW),2), floored_percentage(np.std(LDW),2)) + '\n')
    testfile.write(" LOL&%s pm%s &" % (floored_percentage(np.mean(LOL),2), floored_percentage(np.std(LOL),2)) + '\n')
    testfile.write(" LAL&%s pm%s &" % (floored_percentage(np.mean(LAL),2), floored_percentage(np.std(LAL),2)) + '\n')
    testfile.write(" OALL&%s pm%s &" % (floored_percentage(np.mean(OALL),2), floored_percentage(np.std(OALL),2)) + '\n')
    testfile.write(" OLL&%s pm%s &" % (floored_percentage(np.mean(OLL), 2), floored_percentage(np.std(OLL), 2)) + '\n')
    testfile.write(" Tota&%s pm%s &" % (floored_percentage(np.mean(Tota), 2), floored_percentage(np.std(Tota), 2)) + '\n')
    testfile.write(" Sumw&%s pm%s &" % (floored_percentage(np.mean(Sumw), 2), floored_percentage(np.std(Sumw), 2)) + '\n')
    testfile.write(" SumW&%s pm%s &" % (floored_percentage(np.mean(SumW), 2), floored_percentage(np.std(SumW), 2)) + '\n')
    testfile.write(
        " Lone&%s pm%s &" % (floored_percentage(np.mean(Lone), 2), floored_percentage(np.std(Lone), 2)) + '\n')

    testfile.close()

if __name__ == '__main__':
    Parallel(n_jobs=10)(delayed(mcode)(ite=i) for i in range(10))