import os
import sys
import re
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


fin_path = "/Users/beiyulin/Desktop/atmo2Data_test.al"
fout_path = "/Users/beiyulin/Desktop/atmo2Data_test_Rewrite.al"

fin_orig = open(fin_path, 'rU')
data_orig = fin_orig.readlines()
fin_orig.close()


start_date = datetime.strptime("2018-04-09", '%Y-%m-%d')
f_oneDay = open(fout_path, 'w')

for i in range(0, len(data_orig)):
	l_split = re.split(r' ', data_orig[i])
	date_form = datetime.strptime(l_split[0], '%Y-%m-%d')
	# print("date_form.date()", date_form.date(), start_date, date_form.date() == start_date)
	if (date_form.date() == start_date):
		f_oneDay.write(data_orig[i])
f_oneDay.close()

