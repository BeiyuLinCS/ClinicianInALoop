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
from B_extractData import extractData_func

# finpath = "/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned"
# foutpath = "/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned/WalkingSpeed"


def WalkingSpeed_func(home_name, start_date):

	extractData_func(home_name, start_date)
	
	finpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/raw_shdataset.al" %home_name
	foutpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/perDay.al" %home_name


	# file_name = sys.argv[1:2]
	# print "file_name", file_name

	# file_name = "/raw_shdataset.al"
	### python WalkingSpeed.py /hh101data.al
	### python WalkingSpeed.py /hh102Dec2011_data.al  

	### python WalkingSpeed.py /tm001SepOctdata.al

	# Bed_sensor_name = ['BedroomABed']
	Bed_sensor_name =['BedroomABed'] ##hh102

	# Bath_sensor_names = ['BathroomA']
	Bath_sensor_names = ["BathroomAArea", "BathroomASink","BathroomAToilet"]#['L003', 'MA013', 'D004'] ##hh102

	# Bed_sensor_name = ['M012'] ## hh101
	# Bath_sensor_names = ['MA015', 'D003'] ## hh101

	fin = open(finpath, 'rU')
	data = fin.readlines()
	fin.close()

	dur = 0


	fout = open(foutpath, 'w')

	threshold = 3*60
	low_boundry = 3

	for i in range(0, len(data)-1):
		curr_line = re.split(r'\t', data[i])
		next_line = re.split(r'\t', data[i+1])

		next_line[1] = next_line[1].strip()

		#if curr_line[1].strip() in Bed_sensor_name and  next_line[1][0:9] in Bath_sensor_names:
		# if curr_line[1].strip() in Bed_sensor_name:
		# 	print curr_line
		# 	print next_line

		
		if (curr_line[1].strip() in Bed_sensor_name) and  (next_line[1] in Bath_sensor_names):
			print ((curr_line[1].strip() in Bed_sensor_name), (next_line[1] in Bath_sensor_names))
			print ("")
			curr_line[0] = datetime.strptime(curr_line[0], '%Y-%m-%d %H:%M:%S.%f')
			next_line[0] = datetime.strptime(next_line[0], '%Y-%m-%d %H:%M:%S.%f')

			dur = (next_line[0] - curr_line[0])
			dur_second = dur.total_seconds()
			print("low_boundry, dur_second, threshold", low_boundry, dur_second, threshold)
			print(low_boundry<= dur_second and  dur_second - threshold <= 0)
			print("")
			if low_boundry<= dur_second and  dur_second - threshold <= 0: 
				fout.write(str(curr_line[0]) + "\t" + str(next_line[0]) + "\t" + str(dur) +"\t" + str(dur_second) + "\n")

				### see if same day or not


				# if dur.days == 0:
				# 	dur_per_day.append(dur_second)
					## print "not in the same day", str(curr_line[0]) + "\t" + str(next_line[0]) + "\t" + str(dur) +"\t" + str(dur_second)
					## Checked that all in the same day. 
	fout.close()








