from os import walk
import fnmatch
import os
import re
import numpy as np
from math import floor
def floored_percentage(val, digits):
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}\%\pm '.format(digits, floor(val) / 10 ** digits)
pattern = 'NRFIDH10.50000*'

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
once = []
a1=0
a2=0
a3=0
a4=0
a5=0
a6=0
a7=0
for file in myfiles:
    searchfile = open(file, "r")
    for line in searchfile:


        if "RFSVM" in line:
            l = re.findall("\d+\.\d+", line)
            RFSVM.append(float(l[0]))
        if "RFDIS " in line:
            l = re.findall("\d+\.\d+", line)
            RFDIS.append(float(l[0]))

        if "LATERF&" in line:
            l = re.findall("\d+\.\d+", line)
            LRF.append(float(l[0]))
        if "LATERFDIS" in line:
            l = re.findall("\d+\.\d+", line)
            LRFDIS.append(float(l[0]))
        if "LATEpro" in line:
            l = re.findall("\d+\.\d+", line)
            prorf.append(float(l[0]))
            a8 = float(l[0])/100
        if "LATEprodis" in line:
            l = re.findall("\d+\.\d+", line)
            prorfdis.append(float(l[0]))
            a9 = float(l[0])/100


    searchfile.close()
print(len(RFSVM),len(RFDIS),len(LRF),len(LRFDIS))
print("$"+"%.2f" % np.mean(RFSVM)+r"\%\pm"+"%.2f" % np.std(RFSVM)+"$")
print("&")
print("$"+"%.2f" % np.mean(RFDIS)+r"\%\pm"+"%.2f" % np.std(RFDIS)+"$")
print("&")
print("$"+"%.2f" % np.mean(LRF)+r"\%\pm"+"%.2f" % np.std(LRF)+"$")
print("&")
print("$"+"%.2f" % np.mean(LRFDIS)+r"\%\pm"+"%.2f" % np.std(LRFDIS)+"$")
