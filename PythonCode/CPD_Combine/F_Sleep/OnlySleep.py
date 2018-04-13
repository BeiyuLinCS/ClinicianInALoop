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

def OnlySleepData_func(home_name):
 
	# finpath = "/Users/beiyulin/Desktop/ClinicianSleep/"
	# finpath = "/Users/beiyulin/Desktop/Clinician3/round2_updated/"
	# file_name = "hh102"
	# fin_file = finpath + file_name +"/"+file_name +"Time_Activity_Interval.txt"

	directory_sleep = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/" %home_name

	try:
	    os.makedirs(directory_sleep)
	except OSError as e:
	    if e.errno != errno.EEXIST:
	        raise

	fin_file = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayTime_Activity_Interval.txt" %home_name
	foutpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/Only_Sleep_Data.txt" %home_name
	
	# fou1_file = foutpath + file_name + "/Accumulated_Daily_DayTime_Sleep.txt"
	# fou2_file = foutpath + file_name + "/Accumulated_Daily_NightTime_Sleep.txt"
	# fou3_file = foutpath + file_name + "/Accumulated_Daily_Total_Sleep.txt"

	fin = open(fin_file, 'rU')
	data = fin.readlines()
	fin.close()


	fout_sleep =foutpath 
	fout_s = open(fout_sleep, 'w')

	for i in range(len(data)):
		curr_l_split = re.split(r'\t', data[i])
		
		if curr_l_split[3].strip() == "Sleep":
			fout_s.write(data[i])

	fout_s.close()





