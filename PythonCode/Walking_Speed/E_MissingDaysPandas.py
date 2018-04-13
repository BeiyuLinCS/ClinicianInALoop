import pandas as pd
from pandas.tseries.offsets import *
import os
import numpy as np
import sys
import errno
from datetime import datetime
from datetime import date, timedelta
from D_PerDayOrNot import PerDayOrNot_func 
import re
def Walking_Speed_func(start_date, home_name):

	home_name = sys.argv[1]
	# now = datetime.datetime.now()
	yesterday = date.today() - timedelta(2)  ##datetime.date(2018, 4, 5)
	start_date = yesterday
	start_date_string = datetime.strftime(start_date, "%Y-%m-%d")

	directory_WS = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/" %home_name
	directory_raw_data = '/net/files/home/blin/NewTranslateDirectory/%s/' %home_name
	try:
	    os.makedirs(directory_WS)
	    os.makedirs(directory_raw_data)
	except OSError as e:
	    if e.errno != errno.EEXIST:
	        raise

	PerDayOrNot_func(home_name, start_date)
	
	#data = pd.read_table('/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned//WalkingSpeed/hh101/hh101data.alperDaySpeed', delim_whitespace=True, names=('Date', 'Speed'))
	data = pd.read_table('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/DailySpeed.al' %home_name, delim_whitespace=True, names=('Date', 'Number'))
	# data = pd.read_table('/Users/beiyulin/Desktop/Clinician3/round2_updated/tm002/WS/Accumulated_Daily_Total_Sleep.txt', delimiter=",", names=('Date', 'Number'))
	#print "data", data
	df = pd.DataFrame(data).set_index('Date')

	print "duplicated index", df[df.index.duplicated()]

	df.index = pd.to_datetime(df.index)
	res = df.reindex(pd.date_range(start_date_string, start_date_string, freq=Day()))
	# print res

	#np.savetxt(r'/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/GeronTechnology/ClinicianProject/hh101MatchedDuration/Daily/InsertedMissingDays/Bed_Toilet_Transition_NoMissingDay.txt', res, index = True, fmt='%d')

	res.to_csv('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/NoMissing_Date_DailySpeed.al' %home_name, index=True, header=False)


	fin_path = '/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/NoMissing_Date_DailySpeed.al' %home_name
	fin = open(fin_path, 'rU')
	f_data = fin.readlines()
	data_last = f_data[-1]
	data_last_split = re.split(r",", data_last)

	fout_all_features_path = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/Daily_Updated_tms.txt"
	fout_all_features = open(fout_all_features_path, 'aw')
	fout_all_features.write(data_last_split[1].strip()+ "\n")
	fout_all_features.close()

	
