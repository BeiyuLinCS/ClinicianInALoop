#!/usr/bin/env python
import sys
import time
import string
import calendar
from decimal import *
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import numpy as np
import csv
#import pylab
#import matplotlib.dates as mdates
import calendar
import datetime
from datetime import datetime
from pytz import timezone
import pytz
import re
import os, errno
from dateutil import tz
from collections import Counter
import glob

def OnlyBTTData_func(home_name):
 
	# finpath = "/Users/beiyulin/Desktop/ClinicianSleep/"
	# finpath = "/Users/beiyulin/Desktop/Clinician3/round2_updated/"
	# file_name = "hh102"
	# fin_file = finpath + file_name +"/"+file_name +"Time_Activity_Interval.txt"

	directory_BTT = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/BTT/" %home_name

	try:
	    os.makedirs(directory_BTT)
	except OSError as e:
	    if e.errno != errno.EEXIST:
	        raise

	fin_file = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayTime_Activity_Interval.txt" %home_name
	fout_BTT = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/BTT/OnlyBTT_data.txt" %home_name

	fin = open(fin_file, 'rU')
	data = fin.readlines()
	fin.close()
	
	fout_s = open(fout_BTT, 'w')

	for i in range(len(data)):
		curr_l_split = re.split(r'\t', data[i])
		
		if curr_l_split[3].strip() == "Bed_Toilet_Transition":
			fout_s.write(data[i])

	fout_s.close()





