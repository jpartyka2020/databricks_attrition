# Databricks notebook source
from datetime import datetime, timedelta
import os
import sys

# COMMAND ----------

prediction_file_path = 'dbfs:/FileStore/prediction_output/fortnightly_prediction_files/'
log_file_path = 'dbfs:/FileStore/prediction_output/fortnightly_log_files/'

#Let's start by managing prediction files, deleting any files >= 2 weeks old

#read prediction files and extract the dates_times in their name
prediction_file_list = dbutils.fs.ls(prediction_file_path)

now = datetime.now()

delete_prediction_file_list = []

print("MARK OLD PREDICTION FILES FOR DELETION")
print("================")

for one_file in prediction_file_list:

    this_file_name = one_file.name
    print("Current file: " + prediction_file_path + this_file_name)

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

    # Define a timedelta representing 2 weeks
    two_weeks = timedelta(days=14)
    
    if time_difference >= two_weeks:
        print("mark for deletion")
        delete_prediction_file_list.append(prediction_file_path + this_file_name)
    else:
        print("do not delete just yet")


delete_log_file_list = []
log_file_list = dbutils.fs.ls(log_file_path)

print()
print()
print("MARK OLD LOG FILES FOR DELETION")
print("================")

for one_file in log_file_list:

    this_file_name = one_file.name
    print("Current file: " + log_file_path + this_file_name)

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

    # Define a timedelta representing 2 weeks
    two_weeks = timedelta(days=14)
    
    if time_difference >= two_weeks:
        print("mark for deletion")
        delete_log_file_list.append(log_file_path + this_file_name)
    else:
        print("do not delete just yet")

print()
print()
print("DELETE OLD PREDICTION FILES")
print("================")

for this_delete_file_name in delete_prediction_file_list:
    status_code = dbutils.fs.rm(this_delete_file_name)

    if status_code == True:
        print("delete file: " + this_delete_file_name)

print()
print()
print("DELETE OLD LOG FILES")
print("================")

for this_delete_file_name in delete_log_file_list:
    status_code = dbutils.fs.rm(this_delete_file_name)

    if status_code == True:
        print("delete file: " + this_delete_file_name)

