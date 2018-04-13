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
import glob
# from OnlySleepData import OnlySleepData_func



def same_day_calculation(input_list):

	day_time = 0.0
	night_time = 0.0
	total_time = 0.0
	the_next_day = []

	print input_list

	for i in range(0, len(input_list)):

		start_t = input_list[i][0]
		end_t = input_list[i][1]

		if start_t.day == end_t.day:

			if 8 < int(start_t.hour) < 22:
				day_time = day_time + (end_t - start_t).total_seconds()

			else: 
				night_time = night_time + (end_t - start_t).total_seconds()

		### the case of start 23:50; end at 1:10 am the next day
		if start_t.day != end_t.day: 
			temp_date_time = datetime.strptime(str(start_t.year)+"-"+str(start_t.month)+"-"+str(start_t.day)+ " 23:59:59", '%Y-%m-%d %H:%M:%S')
			night_time = night_time + (temp_date_time - start_t).total_seconds()

			temp_next_day = datetime.strptime(str(end_t.year)+"-"+str(end_t.month)+"-"+str(end_t.day)+ " 00:00:00", '%Y-%m-%d %H:%M:%S')
			the_next_day.append(temp_next_day)
			the_next_day.append(end_t)

	curr_date = str(start_t.year)+"-"+str(start_t.month)+"-"+str(start_t.day)
	total_time = day_time + night_time

	return curr_date, day_time/(60.0*60.0), night_time/(60.0*60.0), total_time/(60.0*60.0), the_next_day



def Sleep_func(home_name):

	# OnlySleepData_func()

	# finpath = "/Users/beiyulin/Desktop/ClinicianSleep/"
	fin_file = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/Only_Sleep_Data.txt" %home_name
	fou1_file = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/Accumulated_Daily_DayTime_Sleep.txt" %home_name
	fou2_file = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/Accumulated_Daily_NightTime_Sleep.txt" %home_name
	fou3_file = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/Sleep/Accumulated_Daily_Total_Sleep.txt" %home_name 

	fin = open(fin_file, 'rU')
	data = fin.readlines()
	fin.close()


	i = 0 

	fout_dayTime = open(fou1_file, 'w')
	fout_nightTime = open(fou2_file, 'w')
	fout_totalTime = open(fou3_file, 'w')

	curr_date_time = []
	temp_len_2 = []

	if (len(data) == 2): 
		curr_l_split2 = re.split(r'\t', data[0])
		next_l_split2 = re.split(r'\t', data[1])
		
		curr_start_date_time2 = datetime.strptime(curr_l_split2[2], '%Y-%m-%d %H:%M:%S')
		curr_end_date_time2 = datetime.strptime(next_l_split2[2], '%Y-%m-%d %H:%M:%S')
		temp_len_2.append([curr_start_date_time2, curr_end_date_time2])

		curr_date2, day_time2, night_time2, total_time2, the_next_day2 = same_day_calculation(temp_len_2)
		print ""
		fout_dayTime.write(curr_date2 + " " + str(round(day_time2, 1)) + '\n')
		fout_nightTime.write(curr_date2 + " " + str(round(night_time2, 1)) + '\n')
		fout_totalTime.write(curr_date2 + " " + str(round(total_time2, 1)) + '\n')
	

	while( i < len(data)-2):
		curr_l_split = re.split(r'\t', data[i])
		next_l_split = re.split(r'\t', data[i+1])

		# print l_split
		# print l_split[2]
		# print ""
		# ['end', '1490080156.0', '2017-03-21 00:09:16', 'Other_Activity\n']
		# 2017-03-21 00:09:16


		if curr_l_split[3].strip() == "Sleep":

			curr_start_date_time = datetime.strptime(curr_l_split[2], '%Y-%m-%d %H:%M:%S')
			curr_end_date_time = datetime.strptime(next_l_split[2], '%Y-%m-%d %H:%M:%S')
			# print curr_start_date_time, curr_end_date_time
			
			curr_date_time.append([curr_start_date_time, curr_end_date_time])

			j = i+2
			next_sleep_start_l = re.split(r"\t", data[j]) ## this is not sleep data
			next_sleep_end_l = re.split(r"\t", data[j+1])
			next_start_date_time = datetime.strptime(next_sleep_start_l[2], '%Y-%m-%d %H:%M:%S')
			next_end_date_time = datetime.strptime(next_sleep_end_l[2], '%Y-%m-%d %H:%M:%S')

			# print "curr_date_time1", curr_date_time

			while (next_start_date_time.day == curr_start_date_time.day):

				print next_sleep_start_l, next_sleep_end_l

				curr_date_time.append([next_start_date_time, next_end_date_time])
				j = j+2
				if j < len(data)-1:
					next_sleep_start_l = re.split(r"\t", data[j])  ### might need some restriction on the j value
					next_sleep_end_l = re.split(r"\t", data[j+1])  ### might need some restriction on the j value
					next_start_date_time = datetime.strptime(next_sleep_start_l[2], '%Y-%m-%d %H:%M:%S')
					next_end_date_time = datetime.strptime(next_sleep_end_l[2], '%Y-%m-%d %H:%M:%S')
					# print "curr_date_time2", curr_date_time
				else:
					break
				


			curr_date, day_time, night_time, total_time, the_next_day = same_day_calculation(curr_date_time)
			# print curr_date, day_time, night_time, total_time, the_next_day
			print ""
			fout_dayTime.write(curr_date + " " + str(round(day_time, 1)) + '\n')
			fout_nightTime.write(curr_date + " " + str(round(night_time, 1)) + '\n')
			fout_totalTime.write(curr_date + " " + str(round(total_time, 1)) + '\n')

			curr_date_time = []

			if the_next_day != []:
				
				# temp_next_day = datetime.strptime(str(end_t.year)+"-"+str(end_t.month)+"-"+str(end_t.day+1)+ " 00:00:00", '%Y-%m-%d %H:%M:%S')
				# the_next_day.append([temp_next_day, end_t])
				temp_next_day = the_next_day[0]
				end_t = the_next_day[1]

				if the_next_day[1].day == next_start_date_time.day:
					curr_date_time.append(the_next_day)
					
				else:
					# print t.strftime('%Y-%m-%d')
					fout_nightTime.write(str(end_t.year) + "-" + str(end_t.month) + "-" + str(end_t.day)+ ' ' + str(round(((end_t - temp_next_day).total_seconds())/(60.0*60.0),1)) + '\n')
					fout_totalTime.write(str(end_t.year) + "-" + str(end_t.month) + "-" + str(end_t.day)+ ' ' + str(round(((end_t - temp_next_day).total_seconds())/(60.0*60.0), 1)) + '\n')
			i = j
		else:
			i += 1					

	fout_dayTime.close()
	fout_nightTime.close()
	fout_totalTime.close()



			





