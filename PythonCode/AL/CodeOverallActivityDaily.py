#!/usr/bin/env python
import sys
import time
import string
import calendar
from decimal import *
import numpy as np
import csv
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
from CleanCalActivityLevel import CleanCalActivityLevel_func

def CodeOverallActivityDaily_func(start_date, home_name):

	CleanCalActivityLevel_func(start_date, home_name)

	finpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/AL/HourlyOverallActivityLevelAL.txt" %home_name
	foutpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/AL/DailyOverallActivityLevelOverall_AL.txt" %home_name


	fin = open(finpath, 'rU')
	data = fin.readlines()
	fin.close()

	AL = []

	fout = open(foutpath, 'w')

	for i in range(0, len(data)-1):
		curr_line = re.split(r'\t', data[i])
		next_line = re.split(r'\t', data[i+1])

		curr_line[0] = datetime.strptime(curr_line[0], '%Y-%m-%d %H:%M:%S')
		next_line[0] = datetime.strptime(next_line[0], '%Y-%m-%d %H:%M:%S')

		##### 2012-07-18 12:00:00	12

		if curr_line[0].day == next_line[0].day :

			if i != len(data) -2:
				AL.append(int(curr_line[1].strip()))
			elif i == len(data) -2:
				AL.append(int(curr_line[1].strip()))
				AL.append(int(next_line[1].strip()))
				fout.write(str(curr_line[0].year)+'-'+str(curr_line[0].month) + "-" + str(curr_line[0].day) +'\t' +str(np.sum(AL)) + '\n')
				AL = []
		else:
			AL.append(int(curr_line[1].strip()))
			fout.write(str(curr_line[0].year)+'-'+str(curr_line[0].month) + "-" + str(curr_line[0].day) +'\t' +str(np.sum(AL)) + '\n')
			AL = []



