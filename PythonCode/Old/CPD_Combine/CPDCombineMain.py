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
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
from datetime import date, timedelta
## creat __init__.py in the test_subfolder directory. 
sys.path.insert(0, "/Users/beiyulin/Desktop/Combine_Code/CPD_Combine")
from A_GenerateNewTranslate.newTranslate import newTranslate_func
from B_FeatureExtraction.FeatureExtract import FeatureExtract_func
from C_ChangPointDetection.OnlineCPD import Online_CPD_Func
from D_InterEventTime.InterEventTime import InterEventTime_func

from F_Sleep.OnlySleep import OnlySleepData_func
from F_Sleep.Sleep import Sleep_func
from F_Sleep.MissingDaysPandas import no_missing_sleep_date_func

from E_BTT.OnlyBTT import OnlyBTTData_func
from E_BTT.changeBedToiletFormat import changeBedToiletFormat_func
from E_BTT.BedToilettm001TestDuration import BedToilettm001TestDuration_func
from E_BTT.MissingDaysPandas import No_Missing_BTT_func

if __name__ == "__main__":

	home_name = sys.argv[1]
	# date = datetime.datetime.now()
	yesterday = date.today() - timedelta(2)  ##datetime.date(2018, 4, 5)
	start_date = yesterday

	newTranslate_func(home_name) ## A: create NewTranslate file under the path: "/net/files/home/blin/NewTranslateDirectory/home_name/" 
	FeatureExtract_func(start_date, home_name) ## B: extract features for the CPD calculation. 
	Online_CPD_Func(home_name)
	InterEventTime_func(home_name)

	## sleep calculation ##
	OnlySleepData_func(home_name)
	Sleep_func(home_name)
	no_missing_sleep_date_func(start_date, home_name)

	## BTT calculation ##
	OnlyBTTData_func(home_name)
	changeBedToiletFormat_func(home_name)
	BedToilettm001TestDuration_func(home_name)
	No_Missing_BTT_func(start_date, home_name)

	

	''' saved under
	write_Matched_path = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayMatched_Activity_CP.txt" %home_name
	write_Matched_Time_Act = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayTime_Activity.txt" %home_name
	write_Matched_Time_Act1 = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/One_dayTime_Activity_Interval.txt" %home_name
	write_duration_out_path = "/net/files/home/blin/PopulationModelling/ExtractedFeatures/EFGapTimeLabel/%s/MatchedDuration/" %home_name
	'''


	