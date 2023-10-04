# Databricks notebook source
from datetime import datetime, timedelta
import os
import sys

# COMMAND ----------

file_path = 'dbfs:/FileStore/prediction_output/rolling_prediction_files/'

#read log files and extract the dates_times in their name
prediction_file_list = dbutils.fs.ls(file_path)

now = datetime.now()

delete_file_list = []

for one_file in prediction_file_list:

    this_file_name = one_file.name
    print("Current file: " + file_path + this_file_name)

    #extract datetime info from this_file_name
    this_file_name_parts_list = this_file_name.split("_")

    print(this_file_name_parts_list)

    #we need to split the h:m:s part of the datetime by colon
    this_datetime_parts_list = this_file_name_parts_list[-1].split(":")

    this_year = int(this_file_name_parts_list[3])
    this_month = int(this_file_name_parts_list[4])
    this_day = int(this_file_name_parts_list[5])

    this_hour = int(this_datetime_parts_list[0])
    this_minute = int(this_datetime_parts_list[1])

    this_file_datetime_obj = datetime(this_year, this_month, this_day, this_hour, this_minute)
    
    # Calculate the time difference between now and this_file_datetime_obj
    time_difference = now - this_file_datetime_obj

    print("Time difference between " + str(now) + " and " + str(this_file_datetime_obj) + " is: " + str(time_difference))

    # Define a timedelta representing 30 minutes
    thirty_minutes = timedelta(minutes=30)
    
    if time_difference >= thirty_minutes:
        print("mark for deletion")
        delete_file_list.append(file_path + this_file_name)
    else:
        print("do not delete just yet")

print("Time to actually delete the files.....")

#delete all files marked for deletion
for this_delete_file_name in delete_file_list:
    status_code = dbutils.fs.rm(this_delete_file_name)

    if status_code == True:
        print("delete file: " + this_delete_file_name)
