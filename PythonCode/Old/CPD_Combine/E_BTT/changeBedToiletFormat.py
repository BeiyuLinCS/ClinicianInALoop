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

def changeBedToiletFormat_func(home_name):

	finpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/BTT/OnlyBTT_data.txt" %home_name
	foutpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/BTT/BedToilet_duration_format.txt" %home_name


	fin = open(finpath, 'rU')
	data = fin.readlines()
	fin.close()

	fout = open(foutpath, 'w')

	for i in range(0, len(data)-1, 2):
		temp_list = []
		curr_l = re.split(r'\t', data[i])
		next_l = re.split(r'\t', data[i+1])
		# print curr_l
		# ['start', '1507291098.0', '2017-10-06 04:58:18', 'Bed_Toilet_Transition\n']
		temp_list.append(float(curr_l[1]))
		temp_list.append(curr_l[2])
		temp_list.append(float(next_l[1]))
		temp_list.append(next_l[2])
		temp_list.append(0.0)
		fout.write(str(temp_list)+"\n")
	fout.close()


