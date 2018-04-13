#!/usr/bin/env python
import sys
import time
import string
import calendar
from decimal import *
import numpy
import numpy as np
import csv
import calendar
import datetime
from datetime import datetime
from pytz import timezone
import pytz
import re
import os 
#import ocpd as oncd
#import onlineTwoArray as oncd
#import linear as oncd
import LL as oncd
from functools import partial
import itertools



def Online_CPD_Func(home_name):
    Desktop_b = False
    if Desktop_b:
        finpath = "/Users/BeiyuLin/Desktop/ExtractedFeatureGlobalTest.txt"
        foutpath = "/Users/BeiyuLin/Desktop/CP.txt"
    
    else:
        # filename1, filename2 = sys.argv[1:3]
        # /tm002/1EF.txt /tm002/1CP.txt
        finpath = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayEF.txt" %home_name 
        print("read in file in Change Point Calculation", home_name)
        foutpath = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayCP.txt" %home_name 
        print("write the change points into file:", home_name)

    data_in = np.array([], dtype = np.float64)
    count = 0

    f = open(finpath, 'rU')
    f_lines = f.readlines()
    count = len(f_lines)
    f.close()
    print("count", count)
    data = [[0]]*(count)

    for i in range(0, count):
        l_split = re.split('\t', f_lines[i])
        l_split[-1] = l_split[-1].strip()
        tdata = []
        for j in range(0, len(l_split)-1):
            tdata.append(np.round(float(l_split[j]),3))
        data[i] = tdata
    a = np.transpose(data)
    #print("a.shape", a.shape, a.shape[0])  ## ('a.shape', (35, 713), 35)
    # # a = list(itertools.chain.from_iterable(a))
    # # a = np.round(a, decimals=3)
    #print("a[34:37]", a[:,35], len(a[:,35])) #### a[:, number] is for one column
    R, maxes = oncd.online_changepoint_detection(count, a)
    final_score = maxes
    #print("final_score", final_score)
    
    alarm=np.zeros(len(final_score))
    for i in range(1, len(final_score)):
        if (final_score[i]< final_score[i-1]):
            alarm[i-1]=1
    
    
    fout = open(foutpath, 'w')
    for i in range(0, len(alarm)):
    	fout.write(str(alarm[i]) + "\n")
    fout.close()













