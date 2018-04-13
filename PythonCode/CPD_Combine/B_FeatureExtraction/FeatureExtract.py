############################################################
# feature_calculation
# -------------------
# Use AlFeature and features defined in actlearn.feature to
# calculate statistical features with sliding window
############################################################

import os
from Functions import *
import logging
import logging.config
from feature.AlData import AlData
from feature.AlFeature import AlFeature
from feature.lastEventHour import AlFeatureEventHour
from feature.lastEventSeconds import AlFeatureEventSecond
from feature.windowDuration import AlFeatureWindowDuration
from feature.lastDominantSensor import AlFeatureLastDominantSensor
from feature.lastSensorInWindow import AlFeatureEventSensor
from feature.sensorCount import AlFeatureSensorCount
from feature.sensorElapseTime import AlFeatureSensorElapseTime
import sys
from collections import deque
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
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
from datetime import date, timedelta


def FeatureExtract_func(start_date, home_name):
    data = AlData()
    data.load_sensor_translation_from_file("/net/files/home/blin/NewTranslateDirectory/%s/Newtranslate" %home_name)
    fin_entire_date = "/net/files/home/cook/rt/%s/data.al" %home_name
    f_oneDay_date = "/net/files/home/blin/NewTranslateDirectory/%s/OneDay_data.al" %home_name


    ### start reading the One Day Data. 
    fin_orig = open(fin_entire_date, 'rU')
    data_orig = fin_orig.readlines()
    fin_orig.close()

    f_oneDay = open(f_oneDay_date, 'w')

    for i in range(0, len(data_orig)):
        l_split = re.split(r' ', data_orig[i])
        # ['2018-04-09', '23:39:05.398299', 'LivingRoom', 'OFF', 'Other_Activity\n']
        date_form = datetime.strptime(l_split[0], '%Y-%m-%d')
        ## start_date is in the date format.
        if (date_form.date() == start_date):
		f_oneDay.write(data_orig[i])
    f_oneDay.close()
    ### end of reading One Day Data. 

    data.load_data_from_file("/net/files/home/blin/NewTranslateDirectory/%s/OneDay_data.al" %home_name)
    # Some basic statistical calculations
    data.calculate_window_size()
    data.calculate_mostly_likely_activity_per_sensor()


    # Configure Features
    feature = AlFeature()
    # Pass Activity and Sensor Info to AlFeature
    feature.populate_activity_list(data.activity_info)
    feature.populate_sensor_list(data.sensor_info)

    # Add lastEventHour Feature
    feature.featureWindowNum = 1
    feature.add_feature(AlFeatureSensorCount(normalize=False))
    feature.add_feature(AlFeatureWindowDuration("/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayGap.txt" %home_name, normalize=False))
    feature.add_feature(AlFeatureEventHour(normalize=False))
    feature.add_feature(AlFeatureEventSensor(per_sensor=False))
    feature.add_feature(AlFeatureLastDominantSensor(per_sensor=False))
    feature.add_feature(AlFeatureEventSecond(normalize=False))
    feature.add_feature(AlFeatureSensorElapseTime(normalize=False))

    # Calculate Features
    feature.populate_feature_array(data.data)

    #foutpath = root_dictory + "ExtractedFeatureGlobalTest.txt" 
    foutpath ="/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayEF.txt" %home_name
    foutpath1 = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayTimeLabel.txt" %home_name
    fout = open(foutpath, 'w')
    fout1 = open(foutpath1, 'w')

    for i in range(0, len(feature.x)):
        #print(type(feature.x[i]))
        for j in range(0, len(feature.x[i])):
            #print("feature.x[i][j]", feature.x[i][j])
            fout.write(str(feature.x[i][j]) + "\t")
        fout.write(str(feature.y[i]) + "\n")
	fout1.write(str(feature.time[i])+"\t")
	fout1.write(str(feature.get_activity_by_index(feature.y[i]))+"\n")
	
    fout.close()
    fout1.close()
    feature = None
    data = None













