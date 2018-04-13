import pandas as pd
from pandas.tseries.offsets import *
import os
import numpy as np
from Sleep import Sleep_func

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



