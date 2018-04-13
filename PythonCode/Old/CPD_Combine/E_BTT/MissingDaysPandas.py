import pandas as pd
from pandas.tseries.offsets import *
import os
import numpy as np



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
