#! /bin/bash

echo "Starting Overall Activity Level Script"

source /net/files/home/blin/PopulationModelling/ClinicianProject/PythonCode/clinician_virtual_environment/bin/activate

while read home_name; do
	echo "Running home $home_name"

	python /net/files/home/blin/PopulationModelling/ClinicianProject/PythonCode/main.py $home_name
	#python /net/files/home/blin/PopulationModelling/ClinicianProject/PythonCode/AL/MissingDaysPandas.py $home_name
done < /net/files/home/blin/PopulationModelling/ClinicianProject/homes.txt
