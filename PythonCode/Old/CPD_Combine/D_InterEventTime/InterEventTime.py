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


#activity_match_CP = []

def activity_count(index0, index1, activity_match_CP):

	### filter out different activities. 
	activity_list = []
	sorted_counted_act = []
	for i in range(index0, index1+1):
		#print("i, length", i)
		#print("index0", index0)
		#print("index1+1", index1+1)
		# print("activity_match_CP[i-1]", activity_match_CP[i-1])
		line_split = re.split("\t", activity_match_CP[i])
		activity_list.append(line_split[2].strip())

	if (all(x== "Other_Activity" for x in activity_list)):
		return (index1, "Other_Activity")
	else:
		len_a_l = len(activity_list)
		sorted_counted_act = Counter(activity_list).most_common(len_a_l)
		## sorted_counted_act: [(activity, 2), (activity, 2), (activity, 1)]

		if (sorted_counted_act[0][0] == "Other_Activity"):
			majority_index = 1
		else:		
			majority_index = 0 

		value_of_majoirty = sorted_counted_act[majority_index][1]  ## retun the value of the majority label. 
		
		#### the if is for the case: [("Other_Activity", 2), ("Wokr", 2)], then majoirty_index + 1 = 2 = len, it can not do for loop.
		if ((len(sorted_counted_act) == 1) or (len(sorted_counted_act) == 2 and (majority_index + 1 == 2 ))):
			return (index1, sorted_counted_act[majority_index][0])

		else:
			for i in range(majority_index + 1, len(sorted_counted_act)):
				if sorted_counted_act[i][1] == value_of_majoirty:   ## more than one majority label. 
					index1 += 1
					activity_count(index0, index1, activity_match_CP)
				return (index1, sorted_counted_act[majority_index][0]) ## return index1


def InterEventTime_func(home_name):
	a = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel"
	# for file in glob.glob(os.path.join(a, '*')):
	file = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s" %home_name

	file_name = home_name

	# finpath1 = "/" + file_name + "/" + file_name + "CP.txt"
	# finpath2 = "/" + file_name + "/" + file_name + "TimeLabel.txt"
	# write_MatchedResult_path = "/" + file_name + "/" + file_name
	# write_Matched_Time_Act = "/" + file_name + "/" + file_name
	# write_duration_out_path = "/" + file_name + "/" + "MatchedDuration/"

	directory = os.path.dirname("/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/MatchedDuration/" %home_name)

	if not os.path.exists(directory):	
		os.mkdir(directory)

	# finpath1, finpath2, write_Matched_path, write_Matched_Time_Act, write_duration_out_path = a+finpath1, a + finpath2, a + write_MatchedResult_path, a+write_Matched_Time_Act, a + write_duration_out_path

	finpath1 = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayCP.txt" %home_name
	finpath2 = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayTimeLabel.txt" %home_name
	write_Matched_path = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayMatched_Activity_CP.txt" %home_name
	write_Matched_Time_Act = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayTime_Activity.txt" %home_name
	write_Matched_Time_Act1 = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayTime_Activity_Interval.txt" %home_name
	write_duration_out_path = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/MatchedDuration/" %home_name

#	if os.path.isfile(write_Matched_Time_Act):
#		continue
		
	if os.path.isfile(finpath1):

		print "file_name", file_name
		print ""

		#### /atmo2/atmo2CP.txt /atmo2/atmo2TimeLabel.txt /atmo2/
		fin1 = open(finpath1, 'rU')   ## except the first 29 lines; fin1 is the CP file. 
		f1 = fin1.readlines()
		fin1.close()
		
		first_line_f1 = f1[0]
		temp_non_zero_index = 0
		if (int(first_line_f1[0].strip()) == 0 ):
			temp_non_zero_index += 1
			next_line_f1 = f1[temp_non_zero_index]
			while (int(next_line_f1[0].strip()) == 0):
				temp_non_zero_index += 1
				next_line_f1 = f1[temp_non_zero_index]

		fin2 = open(finpath2, 'rU')
		f2 = fin2.readlines()
		fin2.close()

		len1 = len(f1)
		len2 = len(f1)
		len2_test = len(f2)

		
		##########################################################################################
		########### match the 1 values and the 0 values with the time and the activity ###########
		########### and write out to Mached_Activity_CP.txt using f1out ##########################
		##########################################################################################
		activity_match_CP = []
		f1out = open(write_Matched_path, 'w')
		for i in range(temp_non_zero_index, len1):

			l2 = f1[i]  ## CP
			l1 = re.split('\t', f2[i])   ## Time and Label from the data before applying CPD. 
			#f1out.write(l1[2].strip() + "\t" + l2[0] + " " + l2[1] + "\t" + l2[4].strip() + '\n')
			#print("l1", l1)
			#print("l2", l2)
			datetimeT = datetime.fromtimestamp(int(float(l1[0].strip()))).strftime('%Y-%m-%d %H:%M:%S')
			f1out.write(l2[0].strip() + "\t" + l1[0].strip() + '\t' +datetimeT + '\t' +l1[1].strip() + '\n')
			activity_match_CP.append( l2[0].strip() + '\t' + l1[0].strip() + "\t" + l1[1].strip() + "\t" )
		f1out.close()
		len_act_match = len(activity_match_CP)

		index_start = 0
		index_end = 0

		##########################################################################################
		########### find the majority lable in each segment and replace the activity label #######
		########### write the labeled segment with timestamp into a file with f2out ##############
		##########################################################################################
		f2out = open(write_Matched_Time_Act, 'w')
		i = 0 
		CP_activity = ""
		returned_new_end_index = 0

		interval_start_end_list = []
		index_start = None

		while( i < len_act_match -1 ):
			l_cur = re.split("\t", activity_match_CP[i])
			l_next = re.split("\t", activity_match_CP[i + 1])

			if int(l_cur[0]) == 1:
				index_start = i 
				i += 1
				continue

			elif (int(l_cur[0]) == 0 and int(l_next[0]) != 1):
				i += 1
				continue

			elif (int(l_cur[0]) == 0 and int(l_next[0]) == 1):
				index_end = i
				##### ignore the 0s before the first 1. 
				if index_start == None:
					#print("before first 1")
					i = i + 1
					continue
				else:
					#print("here", index_start, index_end)

					## returned_new_end_index, and major activity in that interval. 

					returned_new_end_index, CP_activity = activity_count(index_start, index_end, activity_match_CP)

				l_start = re.split("\t", activity_match_CP[index_start])
				l_end = re.split("\t", activity_match_CP[returned_new_end_index])
				datetimeS = datetime.fromtimestamp(int(float(l_start[1]))).strftime('%Y-%m-%d %H:%M:%S')
				datetimeE = datetime.fromtimestamp(int(float(l_end[1]))).strftime('%Y-%m-%d %H:%M:%S')

				### f2out write in timestamp, datetime corresponding to the stamp, re-labelled major activity. 
				f2out.write(l_start[1] + '\t' + datetimeS + '\t' + CP_activity + '\n')
				f2out.write(l_end[1] + '\t' +datetimeE + '\t'+ CP_activity + '\n')

				index_start = returned_new_end_index + 1
				i = returned_new_end_index + 1
		f2out.close()

		##########################################################################################
		########### write the duration of each labeled activity into file ########################
		##########################################################################################

		f2in_path = open(write_Matched_Time_Act, 'rU')
		f2in = f2in_path.readlines()
		f2in_path.close()

		f3out = open(write_Matched_Time_Act1, 'w')

		i = 0 

		while (i < len(f2in)-1):

			l_split_start = re.split(r'\t', f2in[i])
			l_split_next = re.split(r'\t', f2in[i+1])

			## l_split[0], l_split[1], l_split[2]
			# f2out.write(l_start[1] + '\t' + datetimeS + '\t' + CP_activity + '\n')
			# f2out.write(l_end[1] + '\t' +datetimeE + '\t'+ CP_activity + '\n')

			if l_split_next[2].strip() == l_split_start[2].strip():

				j = i + 1

				while (l_split_next[2].strip() == l_split_start[2].strip()):

					if j < len(f2in)-1:

						j = j + 1
						l_split_next = re.split(r'\t', f2in[j])
					else:
						break

				
				f3out.write("start" + "\t"+ f2in[i]) ## start_line
				f3out.write("end" + "\t"+ f2in[j-1]) ## end_line
				# start_timeStamp = l_split_start[0]
				# end_timeStamp = l_split_end[0]
				i = j 

			else: 
				print "error"
				print f2in[i]

		f3out.close()






















