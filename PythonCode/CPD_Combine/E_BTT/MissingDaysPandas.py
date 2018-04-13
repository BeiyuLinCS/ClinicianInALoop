import pandas as pd
from pandas.tseries.offsets import *
import os
import numpy as np
import re


def No_Missing_BTT_func(start_date, home_name):

	#data = pd.read_table('/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned//WalkingSpeed/hh101/hh101data.alperDaySpeed', delim_whitespace=True, names=('Date', 'Speed'))
	data = pd.read_table('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/BTT/BedToilet_duration_Test.txt' %home_name, delim_whitespace=True, names=('Date', 'Number'))
	#print "data", data
	df = pd.DataFrame(data).set_index('Date')

	print "duplicated index", df[df.index.duplicated()]

	df.index = pd.to_datetime(df.index)
	res = df.reindex(pd.date_range(str(start_date), str(start_date), freq=Day()))
	# print res

	#np.savetxt(r'/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/GeronTechnology/ClinicianProject/hh101MatchedDuration/Daily/InsertedMissingDays/Bed_Toilet_Transition_NoMissingDay.txt', res, index = True, fmt='%d')

	res.to_csv('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/BTT/NoMissing_BedToilet.txt' %home_name, index=True, header=False)



	fin_path = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/BTT/NoMissing_BedToilet.txt" %home_name
	fin = open(fin_path, 'rU')
	f_data = fin.readlines()
	data_last = f_data[-1]
	data_last_split = re.split(r",", data_last)

	fout_all_features_path = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/Daily_Updated_tms.txt"
	fout_all_features = open(fout_all_features_path, 'aw')
	fout_all_features.write(data_last_split[1].strip()+ ",")
	fout_all_features.close()
