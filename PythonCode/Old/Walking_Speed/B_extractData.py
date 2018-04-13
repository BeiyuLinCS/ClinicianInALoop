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
from A_data_from_SHdatabase import extract_func

# finpath = "/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/rightTimePeriod"
# foutpath = "/Users/beiyulin/Library/Mobile Documents/com~apple~CloudDocs/data/Cleaned"

def extractData_func(home_name, start_date):

	extract_func(home_name, start_date)

	finpath = "/net/files/home/blin/NewTranslateDirectory/%s/OneDay_Raw_Sensor_data.al" %home_name
	foutpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/WS/raw_shdataset.al" %home_name

	fin = open(finpath, 'rU')
	data = fin.readlines()
	fin.close()

	bed_bath_sensor_names = ["BathroomAArea", "BathroomASink","BathroomAToilet","BedroomABed"]

	#bed_bath_sensor_names = ['BedroomABed']   ### hh101 ['M012', 'MA015', 'D003']. ## hh102 ['M021', 'L003', 'MA013', 'D004']
	# ## tm001: 
	# motion_sensor = ['Control4-Motion']

	
	#['M021', 'L003', 'MA013', 'D004']   ### hh101 ['M012', 'MA015', 'D003']. ## hh102 ['M021', 'L003', 'MA013', 'D004']
	## tm001: 
	# motion_sensor = ['Control4-Motion']

	# bed_bath_sensor_names = ['M012', 'MA015', 'D003']. ## hh011

	# 

	fout = open(foutpath, 'w')

	for line in data:
		l_split = re.split(r'\|', line)
		#print l_split[4], l_split[9], l_split[6]
		l_split[9] = l_split[9].strip()

		#if (l_split[9] in bed_bath_sensor_names or l_split[9][0:9] in 'BathroomA') and l_split[11].strip() in motion_sensor: 
		if (l_split[9] in bed_bath_sensor_names and (l_split[6].strip() in ("ON", "OFF"))): 
			fout.write(l_split[4].strip() + "\t" + l_split[9].strip() + "\t" + l_split[6].strip() + "\n")
	fout.close()



