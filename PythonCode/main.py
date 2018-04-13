import pandas as pd
from pandas.tseries.offsets import *
import os
import numpy as np
import sys
import errno
from datetime import datetime
from datetime import date, timedelta
import re
from influxdb import InfluxDBClient

from AL.MissingDaysPandas import Daily_Overall_Activity_Level_func
from CPD_Combine.CPDCombineMain import CPD_Sleep_BTT_func
from Walking_Speed.E_MissingDaysPandas import Walking_Speed_func


if __name__ == '__main__':

	home_name = sys.argv[1]
	# now = datetime.datetime.now()
	yesterday = date.today() - timedelta(2)  ##datetime.date(2018, 4, 5)
	start_date = yesterday

	Daily_Overall_Activity_Level_func(start_date, home_name)
	CPD_Sleep_BTT_func(start_date, home_name)
	Walking_Speed_func(start_date, home_name)

	#### write to Grafana.
	client = InfluxDBClient('grafana.ailab.wsu.edu', 8086, 'cil', 'XUmF4Ag4K', 'cildata')

	daily_fin_path = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/Daily_Updated_tms.txt"
	daily_fin = open(daily_fin_path, 'rU')
	daily_data = daily_fin.readlines()
	daily_fin.close()

	last_five_lines = daily_data[-5:]

	## 1std, 1.5std, 0.5std.
	std_dic = { 
	"tm002": {"AL": [0.82419, 1.236286, 0.412095], "BTT": [0.82419, 1.236286, 0.412095], 
	"DaySleep":[0.112066,0.1681,0.056033], "NightSleep": [0.595806, 0.893709, 0.297903], "TotalSleep":[0.598462, 0.897694, 0.299231], 
	"WS": [0.052985, 0.079478, 0.026493]}, 
	"tm003": {"AL": [0.0,0.0,0.0], "BTT": [0.0,0.0,0.0],
	"DaySleep":[0.057722, 0.086583, 0.028861], "NightSleep": [1.389576, 2.084365, 0.694788], "TotalSleep":[1.379918, 2.069877, 0.689959], 
	"WS": [0.014893, 0.022339, 0.007446]},
	"tm005": {"AL": [3.149837, 4.724756, 1.574919], "BTT": [3.149837, 4.724756, 1.574919],
	"DaySleep":[0.098944, 0.148416, 0.049472], "NightSleep": [1.92946, 2.89419, 0.96473], "TotalSleep":[1.932209, 2.898314, 0.966105], 
	"WS": [0.013043, 0.019564, 0.006521]},	
	"tm006": {"AL": [1.457069, 2.185603, 0.728534], "BTT": [3.149837, 4.724756, 1.574919],
	"DaySleep":[0.0,0.0,0.0], "NightSleep": [1.057754, 1.586631, 0.528877], "TotalSleep":[1.057754, 1.586631, 0.528877], 
	"WS": [0.003189, 0.004784, 0.001595]},	
	"tm007": {"AL": [0.67811, 1.017166, 0.339055], "BTT": [0.67811, 1.017166, 0.339055],
	"DaySleep":[0.832906, 1.249359, 0.416453], "NightSleep": [1.750045, 2.625067, 0.875022], "TotalSleep":[1.66997, 2.504955, 0.834985], 
	"WS": [0.003943, 0.005915, 0.001972]}	
	}


	for i in range(0, 5):
		print("i after calculation ##############", i)
		#influx_metric = [{}]
		# 2018-04-10,tm003,1911,,,0.0,0.1,0.021
		l_split = re.split(r',', last_five_lines[i])
		written_date = {"date": l_split[0]}
		written_home = {"home": l_split[1]}

		written_AL = {"AL":l_split[2]}  	## overall activity level
		written_BTT = {"BTT": l_split[3]}    ## bed to toilet transition
		written_DaySleep = {"DaySleep": l_split[4]} 		## Day time sleep
		written_NightSleep = {"NightSleep": l_split[5]}		## Night time sleep
		written_TotalSleep = {"TotalSleep": l_split[6]}		## Total Sleep
		written_WS = {"WS": l_split[7].strip()}				## Walking Speed

		influx_metric = [{
		'measurement': 'sensor_data',
		'time': written_date["date"],
		'tags': {'site': written_home["home"], 'data': 'chart'},
		'fields': {
			'btot': written_BTT["BTT"],
			'actlevel': written_AL["AL"],
			'daysleep': written_DaySleep["DaySleep"],
			'nightsleep': written_NightSleep["NightSleep"],
			'totslsleep': written_TotalSleep["TotalSleep"],
			'speed': written_WS["WS"],
			'1std_btt' : std_dic[written_home["home"]]["BTT"][0],
			'1.5std_btt': std_dic[written_home["home"]]["BTT"][1],
			'.5std_btt': std_dic[written_home["home"]]["BTT"][2],
			'1std_act': std_dic[written_home["home"]]["AL"][0],
			'1.5std_act': std_dic[written_home["home"]]["AL"][1],
			'.5std_act': std_dic[written_home["home"]]["AL"][2],
			'1std_daySleep': std_dic[written_home["home"]]["DaySleep"][0],
			'1.5std_daySleep': std_dic[written_home["home"]]["DaySleep"][1],
			'.5std_daySleep': std_dic[written_home["home"]]["DaySleep"][2],
			'1std_nightSleep': std_dic[written_home["home"]]["NightSleep"][0],
			'1.5std_nightSleep': std_dic[written_home["home"]]["NightSleep"][1],
			'.5std_nightSleep': std_dic[written_home["home"]]["NightSleep"][2],
			'1std_sleep': std_dic[written_home["home"]]["TotalSleep"][0],
			'1.5std_sleep': std_dic[written_home["home"]]["TotalSleep"][1],
			'.5std_sleep': std_dic[written_home["home"]]["TotalSleep"][2],
			'1std_speed': std_dic[written_home["home"]]["WS"][0],
			'1.5std_speed': std_dic[written_home["home"]]["WS"][1],
			'.5std_speed': std_dic[written_home["home"]]["WS"][2]
		}
	}]
	print("beofre client write")
	client.write_points(influx_metric)























