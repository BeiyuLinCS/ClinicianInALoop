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

def twoDays(t_start, t_end, ts, te, foutw, dur):
	
	if t_end - t_start <= 24*60*60:		
		dur.append(float(1))
		foutw.write(str(datetime.fromtimestamp(t_start))+ "\t"+ str(np.sum(dur))+"\n")
		dur = []
		dur.append(float(te - t_end))
	return dur

def BedToilettm001TestDuration_func(home_name):
	
	a = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/BTT/BedToilet_duration_format.txt" %home_name
	b = "/net/files/home/blin/cookinfo/ClinicianProject/DailyUpdates/extracted_data/%s/BTT/BedToilet_duration_Test.txt" %home_name
	data = open(a, 'rU')
	Tlines = data.readlines()
	data.close()

 	#foutpath = finpath1
	# [1342870554.0, '2012-07-21 04:35:54', 1342876109.0, '2012-07-21 06:08:29', 5555.0]
	# [1342940600.0, '2012-07-22 00:03:20', 1342943164.0, '2012-07-22 00:46:04', 2564.0]
	# [1342943166.0, '2012-07-22 00:46:06', 1342952172.0, '2012-07-22 03:16:12', 9006.0]
	dur = []
	fout = open(b, 'w')
	length = len(Tlines)
	for i in range(0, length-1):
		# print "i", i
		templ = re.split(r',', Tlines[i])
		nextl = re.split(r',', Tlines[i+1])


		
		t = float(templ[0][1:])
		t0 = float(templ[2])

		next_t = float(nextl[0][1:])
		next_t0 = float(nextl[2])


		t1 = int(t)/(60*60*24)*(60*60*24) ### daily based on local time. 
		t2 = int(t0)/(60*60*24)*(60*60*24)

		
		next_t1 = int(next_t)/(60*60*24)*(60*60*24) 
		next_t2 = int(next_t0)/(60*60*24)*(60*60*24)



		if i < length-2: 

			if t1 == t2 and t2 == next_t1:
				# print "i here", i, datetime.fromtimestamp(t2)
			
				dur.append(float(1))

			elif t1 == t2 and t2 < next_t1: 
				
				dur.append(float(1))

				fout.write(str((datetime.utcfromtimestamp(t2)).date())+"\t"+str(np.sum(dur)) + "\n")
				dur = []

			elif t1 < t2: 
	
				dur = twoDays(t1, t2, t, t0, fout, dur)
				insert_l = str([t2, datetime.utcfromtimestamp(t2), t2 + 24*60*60, datetime.utcfromtimestamp(t2 + 24*60*60)])
				Tlines[i] = insert_l
				i = i 

		elif i == length-2: 
			# print "last line"
			if t1 == t2:
				if t2 == next_t1:
					if next_t1 == next_t2:
						dur.append(float(1))
						dur.append(float(1))
						# print "dur", dur
						fout.write(str((datetime.utcfromtimestamp(next_t2)).date()) +"\t" + str(np.sum(dur)) + "\n")
						dur = []

					if next_t1 < next_t2:
						# print "last line"
						# print "before dur ****", dur
						dur.append(float(1))
						dur.append(float(1))

						# print "dur ****", dur, np.sum(dur)
						fout.write(str((datetime.utcfromtimestamp(next_t1)).date())+"\t"+str(np.sum(dur)) + "\n")
						dur = []
						# dur = twoDays(next_t1, next_t2, next_t, next_t0, fout, dur)
						if next_t2 - next_t1 <= 24*60*60:		
							dur.append(float(1))	

						# print "dur", dur, dur[0]
						fout.write(str((datetime.utcfromtimestamp(next_t2)).date()) +"\t"+str(dur[0]) + "\n")
						dur = []

				elif t2 < next_t1: 
					dur.append(1)
					fout.write(str((datetime.utcfromtimestamp(t2)).date()) + "\t"+ str(np.sum(dur)) + '\n')
					dur = []

					if next_t1 == next_t2:
						dur.append(1)
						fout.write(str((datetime.utcfromtimestamp(next_t2)).date()) + "\t"+ str(np.sum(dur)) + '\n')
						dur = []

					if next_t1 < next_t2:
						dur = twoDays(t1, t2, t, t0, fout, dur)
						fout.write(str((datetime.utcfromtimestamp(next_t2)).date())+ "\t"+str(dur) + "\n")
						dur = []

			elif t1 < t2:
				dur = twoDays(t1, t2, t, t0, fout, dur)
				if t2 < next_t1:
					fout.write(str((datetime.utcfromtimestamp(t2)).date()) + "\t"+str(t0-t2) + '\n')
					if next_t1 == next_t2: 
						fout.write(str((datetime.utcfromtimestamp(next_t2)).date()) + "\t"+str(next_t0 - next_t) + '\n')
						dur = []
					if next_t1 < next_t2:
						dur = twoDays(next_t1, next_t2, next_t, next_t0, fout, dur)
						fout.write(str((datetime.utcfromtimestamp(next_t2)).date()) + "\t"+ str(dur) + "\n")
						dur = []

				elif t2 == next_t1:
					if next_t1 == next_t2:
						fout.write(str((datetime.utcfromtimestamp(next_t2)).date()) + "\t"+ str((t0-t2) + (next_t0-next_t))+'\n')
						dur = []
					if next_t1 < next_t2:
						dur = twoDays(next_t1, next_t2, next_t, next_t0, fout, dur)
						fout.write(str((datetime.utcfromtimestamp(next_t2)).date()) + "\t"+ str(dur) + "\n")
						dur = []

	fout.close()

