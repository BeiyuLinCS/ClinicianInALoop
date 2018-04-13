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
from CleanOnlyOnOff import CleanOnlyOnOff_func


def CleanCalActivityLevel_func(start_date, home_name):

	CleanOnlyOnOff_func(start_date, home_name)

	#finpath = "/Volumes/Seagate Backup Plus Drive/IAQ_Minute_Data/Atmo9S_Minute/MM/Clean/AfterCompleteImputeSeperated/IAQTimePeriod/EachRoom/Sorted/"
	# finpath = "/Users/beiyulin/Desktop/hh102DecJan/data/CleanOnlyOnOff_hh102DecJan.txt"
	# foutpath = "/Users/beiyulin/Desktop/hh102DecJan/data/"
	#foutpath = "/Volumes/Seagate Backup Plus Drive/IAQ_Minute_Data/Atmo9S_Minute/MM/Clean/AfterCompleteImputeSeperated/IAQTimePeriod/"


	# finpath = "/Users/beiyulin/Desktop/Clinician3/round2_updated/tm007/AL/CleanOnlyOnOff.txt"
	# foutpath = "/Users/beiyulin/Desktop/Clinician3/round2_updated/tm007/AL/"
	finpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/AL/CleanOnlyOnOff.txt" % home_name
	foutpath = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/AL/" % home_name

	def utc_to_local(utc_dt):
		# Hardcode zones:
		from_zone = tz.gettz('UTC')
		to_zone = tz.gettz('US/Pacific')
		return utc_dt.replace(tzinfo=from_zone).astimezone(to_zone).replace(tzinfo=None)

	def local_to_utc(pst_dt):
		# Hardcode zones:
		from_zone = tz.gettz('US/Pacific')
		to_zone = tz.gettz('UTC')
		return pst_dt.replace(tzinfo=from_zone).astimezone(to_zone).replace(tzinfo=None)


	# for filename in glob.glob(os.path.join(finpath, '*.txt')):
	templist = []
	#  	data = open(filename, 'rU')
	data = open(finpath, 'rU')
	tempNameSplit = re.split(r"\/", finpath)
	tname = tempNameSplit[-2]
	print tname
	Tlines = data.readlines()
	lenT = len(Tlines)
	print("lenT", lenT)
	firstLine = Tlines[0]
	lastLine = Tlines[lenT-1]
	data.close()

	tempStrsF = firstLine
	tempF = re.split(r'\t', tempStrsF)
	tempF = tempF[0].strip()
	tempStrsSplitF = re.split(r' ', tempF)

	# ('tempStrsSplitF', ['2011-06-13', '12:29:06.215521', 'WorkArea', 'OFF', 'Other_Activity'], <type 'list'>)
	# ('tempStrsSplitF[0]', '2011-06-13')

	# print ("tempStrsSplitF", tempStrsSplitF, type(tempStrsSplitF))
	# print("tempStrsSplitF[0]", tempStrsSplitF[0])

	#print("tempStrsSplitF[2][:-6]", tempStrsSplitF[2][:-6])

	# print ("tempStrsSplitF[1], tempStrsSplitF[1][:-6]", tempStrsSplitF[1], tempStrsSplitF[1][:-6])

	pst1F = datetime.strptime(tempStrsSplitF[0] + ' ' + tempStrsSplitF[1][:-7],'%Y-%m-%d %H:%M:%S')
	firsttime = time.mktime(pst1F.timetuple())  ### mktime is to change the PST into UTC timeStamp. 

	tempStrsL = lastLine
	tempL = re.split(r'\t', tempStrsL)
	tempL = tempL[0].strip()
	tempStrsSplitL = re.split(r' ', tempL)

	pst1L = datetime.strptime(tempStrsSplitL[0] + ' '+ tempStrsSplitL[1][:-7],'%Y-%m-%d %H:%M:%S')
	lasttime = time.mktime(pst1L.timetuple())  ### mktime is to change the PST into UTC timeStamp. 

	print "lasttime", lasttime, pst1L
	a1 = (int(firsttime)/(60*60))*60*60
	fout = open(foutpath + "HourlyOverallActivityLevel" + tname + '.txt', 'w')
	for lines in Tlines:
		# print("read lines")
		tempStrs = lines
		tempSplit = re.split(r'\t', tempStrs)
		tempSplit = tempSplit[0].strip()

		tempStrsSplit = re.split(r' ', tempSplit)
		# print(tempStrsSplit[0] + ' '+ tempStrsSplit[1][:-7])

		if str(tempStrsSplit[1][-7]) == '.': 
			pst1 = datetime.strptime(tempStrsSplit[0] + ' '+ tempStrsSplit[1][:-7],'%Y-%m-%d %H:%M:%S')
		else: 
			pst1 = datetime.strptime(tempStrsSplit[0] + ' '+ tempStrsSplit[1],'%Y-%m-%d %H:%M:%S')
		utc1 = local_to_utc(pst1)
		timestamp1 = time.mktime(pst1.timetuple())  ### mktime is to change the PST into UTC timeStamp. 
		templist.append([timestamp1, utc1, pst1, tempStrsSplit[3], tempStrsSplit[2]])
	count = 0
	i = 0
	lenL = len(templist)
	print "length of templist", lenL
	#print("templist", templist)

	while((float(templist[i][0]))):
		#print("templist[i][3].strip() == ON", templist[i][3].strip() == "ON", templist[i][3].strip())
		if (a1 > (float(templist[i][0]))):
			a1 = a1 - 60*60
			i = i 
			count = 0 
			continue
		elif ( a1 <= (float(templist[i][0])) < (a1 + 60*60)):
			if (templist[i][3].strip() == "ON"):
				if ((i == (lenL -1))):		
					
					fout.write(str(utc_to_local(datetime.utcfromtimestamp(a1))) + '\t')
					fout.write(str(count) + '\n')
					break
				else: 
					count += 1
					i = i + 1
					continue
			if (templist[i][3].strip() == "OFF"):
				if ((i == (lenL -1))):	
					
					fout.write(str(utc_to_local(datetime.utcfromtimestamp(a1))) + '\t')
					fout.write(str(count) + '\n')
					break
				else: 
					count = count
					i = i + 1
					continue
		fout.write(str(utc_to_local(datetime.utcfromtimestamp(a1))) + '\t')
		fout.write(str(count) + '\n')

		if ((float(templist[i][0])) >= (a1 + 60*60)):
			a1 = a1 + 60*60
			count = 0
			i = i 












