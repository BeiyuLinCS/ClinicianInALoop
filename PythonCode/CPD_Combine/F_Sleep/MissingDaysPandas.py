import pandas as pd
from pandas.tseries.offsets import *
import os
import numpy as np
from Sleep import Sleep_func
import re
# if __name__ == '__main__':

def no_missing_sleep_date_func(start_date, home_name):

	# fout_path_dir = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/" %home_name

	# try:
	#     os.makedirs(directory_home_name)
	#     os.makedirs(directory_AL)
	# except OSError as e:
	#     if e.errno != errno.EXIST:
	#         raise
	

	#data = pd.read_table('/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned//WalkingSpeed/hh101/hh101data.alperDaySpeed', delim_whitespace=True, names=('Date', 'Speed'))
	data = pd.read_table('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/Only_Sleep_Data.txt' %home_name, delim_whitespace=True, names=('Date', 'Number'))
	df = pd.DataFrame(data).set_index('Date')
	print "duplicated index", df[df.index.duplicated()]
	df.index = pd.to_datetime(df.index)
	res = df.reindex(pd.date_range(start_date, start_date, freq=Day()))
	res.to_csv('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/NoMissing_Accumulated_Daily_DayTime_Sleep.txt' %home_name, index=True, header=False)


	#data = pd.read_table('/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned//WalkingSpeed/hh101/hh101data.alperDaySpeed', delim_whitespace=True, names=('Date', 'Speed'))
	data = pd.read_table('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/Accumulated_Daily_NightTime_Sleep.txt' %home_name, delim_whitespace=True, names=('Date', 'Number'))
	df = pd.DataFrame(data).set_index('Date')
	print "duplicated index", df[df.index.duplicated()]
	df.index = pd.to_datetime(df.index)
	res = df.reindex(pd.date_range(start_date, start_date, freq=Day()))
	res.to_csv('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/NoMissing_Accumulated_Daily_NightTime_Sleep.txt' %home_name, index=True, header=False)


	data = pd.read_table('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/Accumulated_Daily_Total_Sleep.txt' %home_name, delim_whitespace=True, names=('Date', 'Number'))
	df = pd.DataFrame(data).set_index('Date')
	print "duplicated index", df[df.index.duplicated()]
	df.index = pd.to_datetime(df.index)
	res = df.reindex(pd.date_range(start_date, start_date, freq=Day()))
	res.to_csv('/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/NoMissing_Accumulated_Daily_Total_Sleep.txt' %home_name, index=True, header=False)

	fin_path1 = '/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/NoMissing_Accumulated_Daily_DayTime_Sleep.txt' %home_name
	fin_path2 = '/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/NoMissing_Accumulated_Daily_NightTime_Sleep.txt' %home_name
	fin_path3 = '/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/NoMissing_Accumulated_Daily_Total_Sleep.txt' %home_name

	## DAY TIME
	fin1 = open(fin_path1, 'rU')
	f_data1 = fin1.readlines()
	data_last1 = f_data1[-1]
	data_last_split1 = re.split(r",", data_last1)

	## Night TIME
	fin2 = open(fin_path2, 'rU')
	f_data2 = fin2.readlines()
	data_last2 = f_data2[-1]
	data_last_split2 = re.split(r",", data_last2)

	## Total TIME
	fin3 = open(fin_path3, 'rU')
	f_data3 = fin3.readlines()
	data_last3 = f_data3[-1]
	data_last_split3 = re.split(r",", data_last3)

	fout_all_features_path = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/Daily_Updated_tms.txt"
	fout_all_features = open(fout_all_features_path, 'aw')
	fout_all_features.write(data_last_split1[1].strip()+ "," + data_last_split2[1].strip() + "," + data_last_split3[1].strip() + ",")
	fout_all_features.close()






