#!/usr/bin/env python
import sys
import time
import string
import calendar
from decimal import *
import numpy
import numpy as np
import csv
import calendar
import datetime
from datetime import datetime
from pytz import timezone
import pytz
import re
import os 
import glob
import errno
from dateutil import tz

def CleanOnlyOnOff_func(start_date, home_name):

	#finpath = "/Volumes/Seagate Backup Plus Drive/IAQ_Minute_Data/Atmo9S_Minute/MM/Clean/AfterCompleteImputeSeperated/IAQTimePeriod/EachRoom/Sorted/"
	# finpath = "/Users/beiyulin/Desktop/hh102DecJan/data/data.al"
	# foutpath = "/Users/beiyulin/Desktop/hh102DecJan/data/"

	# finpath = "/Users/beiyulin/Desktop/Clinician3/Raw_Labeled_Data/tm002/oneYear.txt"
	# finpath = "/Users/beiyulin/Desktop/Clinician3/round2_updated/data/tm007/data.al"
	# foutpath = "/Users/beiyulin/Desktop/Clinician3/round2_updated/tm007/AL/"

	### make a directory of home_name
	## make a directory of AL

	directory_home_name = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/" %home_name
	directory_AL = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/AL/" %home_name

	try:
	    os.makedirs(directory_home_name)
	    os.makedirs(directory_AL)
	except OSError as e:
	    if e.errno != errno.EEXIST:
	        raise
	
	
	finpath = "/net/files/home/cook/rt/%s/data.al" % home_name
	foutpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/AL/CleanOnlyOnOff.txt" % home_name
	
	
	templist = []
	#  	data = open(filename, 'rU')
	data = open(finpath, 'rU')
	tempNameSplit = re.split(r"\/", finpath)
	tname = tempNameSplit[-2]
	print tname
	Tlines = data.readlines()
	lenT = len(Tlines)
	print("lenT", lenT)
	# firstLine = Tlines[0]
	# lastLine = Tlines[lenT-1]
	data.close()

	# tempStrsF = firstLine
	# tempF = re.split(r'\t', tempStrsF)
	# tempF = tempF[0].strip()

	# fout = open(foutpath + "CleanOnlyOnOff_" + tname + '.txt', 'w')
	fout = open(foutpath, 'w')

	for l in Tlines:
		tempStrsF = l
		tempF = re.split(r' ', tempStrsF)
		
		# print "tempF", tempF[0], "tempF[3]", tempF[3] 
		# print "tempF[0]", tempF[0]
		# tempF ['2011-06-14', '10:17:48.011953', 'Kitchen', 'OFF', 'Cook']
		# tempF[0] 2011-06-14
		date_form = datetime.strptime(tempF[0], '%Y-%m-%d')
		if (tempF[3] in ('ON', 'OFF') and date_form.date() == start_date):
			fout.write(' '.join(tempF))







