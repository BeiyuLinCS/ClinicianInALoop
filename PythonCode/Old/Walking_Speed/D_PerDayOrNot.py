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
import os 
from dateutil import tz
from collections import Counter
from datetime import timedelta
from C_WalkingSpeed import WalkingSpeed_func

# finpath = "/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned/WalkingSpeed"
# foutpath = "/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned/WalkingSpeed"

def PerDayOrNot_func(home_name, start_date):
	
	WalkingSpeed_func(home_name, start_date)

	finpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/perDay.al" %home_name
	foutpath1 = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/perDay2.al" %home_name
	foutpath2 = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/DailySpeed.al" %home_name

	# file_name = sys.argv[1:2]
	# print "file_name", file_name

	# file_name = "/raw_shdataset.alperDay"

	distance = 1.76 ## hh101 5.38 ## hh102: 4.30 ## tm001: 1.76
	### python PerDayOrNot.py /hh102Year2012data.alperDay
	### python PerDayOrNot.py /tm001SepOctdata.alperDay
	### python PerDayOrNot.py /hh102Dec2011_data.alperDay

	fin = open(finpath, 'rU')
	data = fin.readlines()
	fin.close()

	dur = 0
	per_day_or_not = 0
	dur_per_day = []

	fout = open(foutpath1, 'w')
	foutSpeed = open(foutpath2, 'w')

	# threshold = datetime.strptime("0000-00-00 00:03:00.000000", '%Y-%m-%d %H:%M:%S.%f')
	threshold = 3*60

	for i in range(0, len(data)-1):
		curr_line = re.split(r'\t', data[i])
		next_line = re.split(r'\t', data[i+1])


		curr_line[1] = datetime.strptime(curr_line[1], '%Y-%m-%d %H:%M:%S.%f')
		next_line[0] = datetime.strptime(next_line[0], '%Y-%m-%d %H:%M:%S.%f')


		# per_day_or_not = (next_line[0] - curr_line[1])	

		### same day
		# print "per_day_or_not", per_day_or_not, per_day_or_not.days

		# if per_day_or_not.days == 0:
		#print "curr_line[1].day, next_line[0].day", curr_line[1].day, next_line[0].day
		if curr_line[1].day == next_line[0].day and i != len(data) -2:
			dur_per_day.append(float(curr_line[3].strip()))
		

		### Not the same day
		else:
			dur_per_day.append(float(curr_line[3].strip()))

			if i == len(data) -2:
				dur_per_day.append(float(next_line[3].strip()))
			
			# print "curr_line[1], next_line[0]", curr_line[1], next_line[0], dur_per_day
			fout.write(str(curr_line[1].year)+'-'+ str(curr_line[1].month)+'-'+str(curr_line[1].day)+'\t'+str(np.sum(dur_per_day)) + "\n")
			foutSpeed.write(str(curr_line[1].year)+'-'+ str(curr_line[1].month)+'-'+str(curr_line[1].day)+'\t'+str(distance/np.sum(dur_per_day)) + "\n")
			dur_per_day = []

			


