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
import glob
import errno
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import re
from datetime import date, timedelta
from CodeOverallActivityDaily import CodeOverallActivityDaily_func 


def Daily_Overall_Activity_Level_func(start_date, home_name):
	
	### get the today's date; strftime
	### then get back to yesterday. today.day -1 

	## python test.py argv[1] argv[2]
	## python argv[0] argv[1] argv[2]

	## run
	## python test.py hh101
	# home_name = sys.argv[1]
	# # now = datetime.datetime.now()
	# yesterday = date.today() - timedelta(2)  ##datetime.date(2018, 4, 5)
	# start_date = yesterday

	CodeOverallActivityDaily_func(start_date, home_name)  ## the start_date, end_date should be in date format. 

	#data = pd.read_table('/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned//WalkingSpeed/hh101/hh101data.alperDaySpeed', delim_whitespace=True, names=('Date', 'Speed'))
	data = pd.read_table('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/AL/DailyOverallActivityLevelOverall_AL.txt' %home_name, delim_whitespace=True, names=('Date', 'Number'))
	# data = pd.read_table('/Users/beiyulin/Desktop/Clinician3/round2_updated/tm002/WS/Accumulated_Daily_Total_Sleep.txt', delimiter=",", names=('Date', 'Number'))
	#print "data", data
	df = pd.DataFrame(data).set_index('Date')

	print "duplicated index", df[df.index.duplicated()]

	df.index = pd.to_datetime(df.index)
	res = df.reindex(pd.date_range(start_date, start_date, freq=Day()))
	# print res

	#np.savetxt(r'/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/GeronTechnology/ClinicianProject/hh101MatchedDuration/Daily/InsertedMissingDays/Bed_Toilet_Transition_NoMissingDay.txt', res, index = True, fmt='%d')

	res.to_csv('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/AL/NoMissing_DailyOverallActivityLevelOverall_AL.txt' %home_name, index=True, header=False)

	fin_path = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/AL/NoMissing_DailyOverallActivityLevelOverall_AL.txt" %home_name
	fin = open(fin_path, 'rU')
	f_data = fin.readlines()
	data_last = f_data[-1]
	data_last_split = re.split(r",", data_last)

	fout_all_features_path = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/Daily_Updated_tms.txt"
	fout_all_features = open(fout_all_features_path, 'aw')
	fout_all_features.write(data_last_split[0] + "," + home_name + ","+ data_last_split[1].strip()+ ",")
	fout_all_features.close()

	# 2017-06-22,,0.0,,,,,0.8241904760602379,	










