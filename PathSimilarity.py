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
    weight =clf.oob_score_ #clf.score(X[test_indices], Y[test_indices])
    #print(1 - clf.oob_score_)
    n_samples = X.shape[0]
    dis = np.zeros((n_samples,n_samples))
    trees = clf.estimators_
    for i in range(n_samples):
        dis[i][i] = 0
    for k in range(len(trees)):
        pa = trees[k].decision_path(X)
        for i in range(n_samples):
            for j in range(i+1,n_samples):
                a = pa[i]
                a = a.toarray()
                a = np.ravel(a)

                b = pa[j]
                b = b.toarray()
                b = np.ravel(b)
                score = a == b
                d = float(score.sum())/len(a)
                dis[i][j] = dis[j][i] = dis[i][j]+1-d
    dis = dis/n_trees
    X_features1 = np.transpose(dis)
    X_features2 = X_features1[train_indices]
    X_features3 = np.transpose(X_features2)
    return X_features3[train_indices],X_features3[test_indices],weight,pred,prob,clf



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
    testfile = open("pathsimilarity%f_%f.txt" % (R,ite), 'w')
    numberofclass = 2
    n_neighbours = 7
    n_views = 5
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
    Y = pandas.read_csv('label_IDHcodel.csv', header=None)
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


    X_features_train1, X_features_test1, w1, pred1, prob1, RFV1 = rf_dis(n_trees=500, X=Xnew1, Y=Y, train_indices=train_indices,
                                                                   test_indices=test_indices, seed=seed)
    print("hello1")
    X_features_train2, X_features_test2, w2, pred2, prob2, RFV2 = rf_dis(n_trees=500, X=Xnew2, Y=Y,
                                                                         train_indices=train_indices,
                                                                     test_indices=test_indices, seed=seed)
    print("hello2")
    X_features_train3, X_features_test3, w3, pred3, prob3, RFV3 = rf_dis(n_trees=500, X=Xnew3, Y=Y,
                                                                         train_indices=train_indices,
                                                                         test_indices=test_indices, seed=seed)
    print("hello3")
    X_features_train4, X_features_test4, w4, pred4, prob4, RFV4 = rf_dis(n_trees=500, X=Xnew4, Y=Y,
                                                                         train_indices=train_indices,
                                                                         test_indices=test_indices, seed=seed)
    print("hello4")
    X_features_train5, X_features_test5, w5, pred5, prob5, RFV5 = rf_dis(n_trees=500, X=Xnew5, Y=Y,
                                                                         train_indices=train_indices,
                                                                         test_indices=test_indices, seed=seed)
    print("hello5")

    W= [w1,w2,w3,w4,w5]
    weight = [max(w1-(1/numberofclass),0),max(w2-(1/numberofclass),0),max(w3-(1/numberofclass),0),max(w4-(1/numberofclass),0),max(w5-(1/numberofclass),0)]
    ww = [1,1,1,1,1]
    pro = [prob1, prob2, prob3, prob4, prob5]
    ave_pro = prob1+ prob2+ prob3+ prob4+prob5
    simplepro = np.zeros((len(test_indices), len(pro)))
    for le in range(len(pro)):
        mprf = pro[le]
        for i in range(mprf.shape[0]):
            simplepro[i, le] = max(mprf[i])
    O_RFlocal = np.zeros((len(test_indices), n_views))
    weightc_RFlocal  = np.zeros((len(test_indices), n_views))
    Wc_RFlocal = np.zeros((len(test_indices), n_views))
    weightp_RFlocal = np.zeros((len(test_indices), n_views))
    Wp_RFlocal = np.zeros((len(test_indices), n_views))
    LocalRFlocal = np.zeros((len(test_indices), n_views))

    M_O_RFlocal = np.zeros((len(test_indices), n_views))
    M_weightc_RFlocal = np.zeros((len(test_indices), n_views))
    M_Wc_RFlocal = np.zeros((len(test_indices), n_views))
    M_weightp_RFlocal = np.zeros((len(test_indices), n_views))
    M_Wp_RFlocal = np.zeros((len(test_indices), n_views))
    M_LocalRFlocal = np.zeros((len(test_indices), n_views))
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
        for ind in range(len(test_indices)):
            nei = XFT[ind].argsort()[:n_neighbours]
            oob = RFV.oob_decision_function_
            la = 0
            for j in range(len(nei)):
                sx = oob[nei[j]]
                sy = Y[train_indices[nei[j]]]
                la = la + sx[sy]
            la = la / n_neighbours
            lla = max(la - 1 / numberofclass, 0)
            O_RFlocal[ind, nv] = la
            weightc_RFlocal[ind, nv] = la * simplepro[ind, nv]  # R7
            Wc_RFlocal[ind, nv] = la * W[nv] * simplepro[ind, nv]  # R9
            weightp_RFlocal[ind, nv] = lla * weight[nv] * max(simplepro[ind, nv] - 1 / numberofclass, 0)  # 11
            Wp_RFlocal[ind, nv] = max(simplepro[ind, nv] - 1 / numberofclass, 0) * lla  # R13
            LocalRFlocal[ind, nv] = max(simplepro[ind, nv] - 1 / numberofclass, 0) * la  # R15

            M_O_RFlocal[ind, nv] = lla
            M_weightc_RFlocal[ind, nv] = max(la * simplepro[ind, nv] - 1 / numberofclass, 0)  # R8
            M_Wc_RFlocal[ind, nv] = lla + weight[nv] + max(simplepro[ind, nv] - 1 / numberofclass, 0)  # 10
            M_weightp_RFlocal[ind, nv] = max(la + weight[nv] - 1 / numberofclass, 0)
            M_Wp_RFlocal[ind, nv] = lla * simplepro[ind, nv]  # R14
            M_LocalRFlocal[ind, nv] = max(simplepro[ind, nv] + la - 2 / numberofclass, 0)  # R16

    #weighted LRF
    resall1 = np.column_stack((pred1, pred2, pred3, pred4, pred5))
    Weighted = list(range(len(test_indices)))
    MV = list(range(len(test_indices)))
    MWeighted = list(range(len(test_indices)))
    LO_RFlocal = list(range(len(test_indices)))
    Lweightc_RFlocal = list(range(len(test_indices)))
    LWc_RFlocal = list(range(len(test_indices)))
    Lweightp_RFlocal = list(range(len(test_indices)))
    LWp_RFlocal = list(range(len(test_indices)))
    LLocalRFlocal = list(range(len(test_indices)))

    LM_O_RFlocal = list(range(len(test_indices)))
    LM_weightc_RFlocal = list(range(len(test_indices)))
    LM_Wc_RFlocal = list(range(len(test_indices)))
    LM_weightp_RFlocal = list(range(len(test_indices)))
    LM_Wp_RFlocal = list(range(len(test_indices)))
    LM_LocalRFlocal = list(range(len(test_indices)))
    AVE = list(range(len(test_indices)))
    #resall11 = np.column_stack((prob1, prob2, prob3, prob4, prob5))
    for i in range(len(test_indices)):
        MV[i] = weightedComb(resall1[i],ww)
        Weighted[i] = weightedComb(resall1[i],W)
        MWeighted[i] = weightedComb(resall1[i], weight)
        LO_RFlocal[i] = weightedComb(resall1[i],O_RFlocal[i])
        LM_O_RFlocal[i] = weightedComb(resall1[i], M_O_RFlocal[i])
        Lweightc_RFlocal[i] = weightedComb(resall1[i], weightc_RFlocal[i])
        LM_weightc_RFlocal[i] = weightedComb(resall1[i], M_weightc_RFlocal[i])
        LWc_RFlocal[i] = weightedComb(resall1[i], Wc_RFlocal[i])
        LM_Wc_RFlocal[i] = weightedComb(resall1[i], M_Wc_RFlocal[i])
        Lweightp_RFlocal[i] = weightedComb(resall1[i], weightp_RFlocal[i])
        LM_weightp_RFlocal[i] = weightedComb(resall1[i], M_weightp_RFlocal[i])
        LWp_RFlocal[i] = weightedComb(resall1[i], Wp_RFlocal[i])
        LM_Wp_RFlocal[i] = weightedComb(resall1[i], M_Wp_RFlocal[i])
        LLocalRFlocal[i] = weightedComb(resall1[i], LocalRFlocal[i])
        LM_LocalRFlocal[i] = weightedComb(resall1[i], M_LocalRFlocal[i])
        AVE[i] = (list(ave_pro[i])).index(max(ave_pro[i]))

    R1 = accuracy_score(Y[test_indices], MV)
    R2 = accuracy_score(Y[test_indices], Weighted)
    R3 = accuracy_score(Y[test_indices], MWeighted)
    R4 = accuracy_score(Y[test_indices], AVE)

    R5 = accuracy_score(Y[test_indices], LO_RFlocal)
    R6 = accuracy_score(Y[test_indices], LM_O_RFlocal)
    R7 = accuracy_score(Y[test_indices], Lweightc_RFlocal)
    R8 = accuracy_score(Y[test_indices], LM_weightc_RFlocal)

    R9 = accuracy_score(Y[test_indices], LWc_RFlocal)
    R10 = accuracy_score(Y[test_indices], LM_Wc_RFlocal)
    R11 = accuracy_score(Y[test_indices], Lweightp_RFlocal)
    R12 = accuracy_score(Y[test_indices], LM_weightp_RFlocal)

    R13 = accuracy_score(Y[test_indices], LWp_RFlocal)
    R14 = accuracy_score(Y[test_indices], LM_Wp_RFlocal)
    R15 = accuracy_score(Y[test_indices], LLocalRFlocal)
    R16 = accuracy_score(Y[test_indices], LM_LocalRFlocal)

    testfile.write(" R1&%s pm%s &" % (floored_percentage(np.mean(R1),2), floored_percentage(np.std(R1),2)) + '\n')
    testfile.write(" R2&%s pm%s &" % (floored_percentage(np.mean(R2),2), floored_percentage(np.std(R2),2)) + '\n')
    testfile.write(" R3&%s pm%s &" % (floored_percentage(np.mean(R3),2), floored_percentage(np.std(R3),2)) + '\n')
    testfile.write(" R4&%s pm%s &" % (floored_percentage(np.mean(R4),2), floored_percentage(np.std(R4),2)) + '\n')
    testfile.write(" R5&%s pm%s &" % (floored_percentage(np.mean(R5), 2), floored_percentage(np.std(R5), 2)) + '\n')
    testfile.write(" R6&%s pm%s &" % (floored_percentage(np.mean(R6), 2), floored_percentage(np.std(R6), 2)) + '\n')
    testfile.write(" R7&%s pm%s &" % (floored_percentage(np.mean(R7), 2), floored_percentage(np.std(R7), 2)) + '\n')
    testfile.write(" R8&%s pm%s &" % (floored_percentage(np.mean(R8), 2), floored_percentage(np.std(R8), 2)) + '\n')
    testfile.write(" R9&%s pm%s &" % (floored_percentage(np.mean(R9), 2), floored_percentage(np.std(R9), 2)) + '\n')
    testfile.write(" R10&%s pm%s &" % (floored_percentage(np.mean(R10), 2), floored_percentage(np.std(R10), 2)) + '\n')
    testfile.write(" R11&%s pm%s &" % (floored_percentage(np.mean(R11), 2), floored_percentage(np.std(R11), 2)) + '\n')
    testfile.write(" R12&%s pm%s &" % (floored_percentage(np.mean(R12), 2), floored_percentage(np.std(R12), 2)) + '\n')
    testfile.write(" R13&%s pm%s &" % (floored_percentage(np.mean(R13), 2), floored_percentage(np.std(R13), 2)) + '\n')
    testfile.write(" R14&%s pm%s &" % (floored_percentage(np.mean(R14), 2), floored_percentage(np.std(R14), 2)) + '\n')
    testfile.write(" R15&%s pm%s &" % (floored_percentage(np.mean(R15), 2), floored_percentage(np.std(R15), 2)) + '\n')
    testfile.write(" R16&%s pm%s &" % (floored_percentage(np.mean(R16), 2), floored_percentage(np.std(R16), 2)) + '\n')
    testfile.close()

if __name__ == '__main__':
    Parallel(n_jobs=10)(delayed(mcode)(ite=i) for i in range(10))