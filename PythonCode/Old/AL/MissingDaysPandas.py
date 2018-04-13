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
from datetime import date, timedelta
from CodeOverallActivityDaily import CodeOverallActivityDaily_func 


if __name__ == '__main__':
	
	### get the today's date; strftime
	### then get back to yesterday. today.day -1 

	## python test.py argv[1] argv[2]
	## python argv[0] argv[1] argv[2]

	## run
	## python test.py hh101
	home_name = sys.argv[1]
	# now = datetime.datetime.now()
	yesterday = date.today() - timedelta(2)  ##datetime.date(2018, 4, 5)
	start_date = yesterday

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


