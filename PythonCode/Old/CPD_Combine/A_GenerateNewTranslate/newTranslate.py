#!/usr/bin/env python
import sys
import re
import os 
import errno

def newTranslate_func(home_name):

    # root_directory = "/net/files/home/blin/cookinfo/update_rt_Feb_2018/"
    # root_directory = "/net/files/home/blin/cookinfo/IAQ_Updated_Date/"

    home_name = sys.argv[1]
    root_directory_home = "/net/files/home/cook/rt/%s/" %home_name
    out_directory_home = "/net/files/home/blin/NewTranslateDirectory/%s/" %home_name

    try:
        os.makedirs(out_directory_home)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    # yesterday = date.today() - timedelta(1)  ##datetime.date(2018, 4, 5)
    # start_date = yesterday
    
    # for dir0 in os.listdir(root_directory):
    # f1name = dir0
    filename = root_directory_home + "translate"
    ## for example: /net/files/home/cook/rt/tm002/translate

    out_filename = out_directory_home + "Newtranslate"

    f_orig = open(filename,"rU")
    fin = f_orig.readlines()

    f_out = open(out_filename, "w")

    for i in range(0, len(fin)):
    	l_split = re.split(' ',fin[i])
    	print l_split[1].strip(), type(l_split[1].strip())
	if (l_split[1].strip() == "Ignore"):
		#print l_split[1].strip(), type(l_split[1].strip())
		continue
	else:
		f_out.write(str(l_split[1].strip()) + "\t")
    		f_out.write(str(l_split[1].strip()) + "\t")
    		f_out.write(str(l_split[1].strip()) + "\n")
    f_out.close()
    f_orig.close()
