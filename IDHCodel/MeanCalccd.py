from os import walk
import fnmatch
import os
import re
import numpy as np
from math import floor
def floored_percentage(val, digits):
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}\%\pm '.format(digits, floor(val) / 10 ** digits)
pattern = '5nonIDH10.700000*'

files = os.listdir('.')
myfiles = []
for name in files:
    if fnmatch.fnmatch(name, pattern):
        myfiles.append(name)
RF = []
RELF = []
SVM = []
RFSVM = []
RFDIS = []
LRF = []
LRFDIS = []
Concatenated = []
for file in myfiles:
    searchfile = open(file, "r")
    for line in searchfile:
        if "AWA 0.1 RF" in line:
            l = re.findall("\d+\.\d+", line)
            RF.append(float(l[1]))

        if "RELF" in line:
            l = re.findall("\d+\.\d+", line)
            RELF.append(float(l[0]))
        if "SVMRFE" in line:
            l = re.findall("\d+\.\d+", line)
            SVM.append(float(l[0]))
        if "RFSVM" in line:
            l = re.findall("\d+\.\d+", line)
            RFSVM.append(float(l[0]))
        if "RFDIS" in line:
            l = re.findall("\d+\.\d+", line)
            RFDIS.append(float(l[0]))
        if "LATERF" in line:
            l = re.findall("\d+\.\d+", line)
            LRF.append(float(l[0]))
        if "LATERFDIS" in line:
            l = re.findall("\d+\.\d+", line)
            LRFDIS.append(float(l[0]))
        if "Concatenated multi view" in line:
            l = re.findall("\d+\.\d+", line)
            Concatenated.append(float(l[0]))


    searchfile.close()
print("$"+"%.2f" % np.mean(RF)+r"\%\pm"+"%.2f" % np.std(RF)+"$")
print("&")
print("$"+"%.2f" % np.mean(RELF)+r"\%\pm"+"%.2f" % np.std(RELF)+"$")
print("&")
print("$"+"%.2f" % np.mean(SVM)+r"\%\pm"+"%.2f" % np.std(SVM)+"$")
print("&")
print("$"+"%.2f" % np.mean(RFSVM)+r"\%\pm"+"%.2f" % np.std(RFSVM)+"$")
print("&")
print("$"+"%.2f" % np.mean(RFDIS)+r"\%\pm"+"%.2f" % np.std(RFDIS)+"$")
print("&")
print("$"+"%.2f" % np.mean(LRF)+r"\%\pm"+"%.2f" % np.std(LRF)+"$")
print("&")
print("$"+"%.2f" % np.mean(LRFDIS)+r"\%\pm"+"%.2f" % np.std(LRFDIS)+"$")
print("$"+"%.2f" % np.mean(Concatenated)+r"\%\pm"+"%.2f" % np.std(Concatenated)+"$")