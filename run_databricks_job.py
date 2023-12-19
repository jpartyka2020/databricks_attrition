import requests
import json
import sys

#use your value of DATABRICKS_TOKEN_STRING
DATABRICKS_TOKEN_STRING = "dapiad60e91111da52a2da2c6b03441f228c"

WORKSPACE_URL = "https://dbc-e12db167-30b0.cloud.databricks.com"

#this maps pod to job ID in databricks
db_name_to_job_id_dict = {}

#this is a list that will determine which jobs we run
db_job_run_list = []

#for all prod databases
db_name_to_job_id_dict['secure1'] = 833279762035356
db_name_to_job_id_dict['secure2'] = 211938150855559
db_name_to_job_id_dict['secure3'] = 1046249980976101
db_name_to_job_id_dict['secure4'] = 477313925068332
db_name_to_job_id_dict['secure5'] = 794041493029182
db_name_to_job_id_dict['staging1'] = 69934995721536
db_name_to_job_id_dict['staging4'] = 780188600233987
db_name_to_job_id_dict['eu1'] = 327857995367657
db_name_to_job_id_dict['eustaging1'] = 710517488266939
db_name_to_job_id_dict['qaintx'] = 551648678160308

#dev job ID, in case you want to run this
#db_name_to_job_id_dict['dev'] = 572811449565005


#if no command-line parameters are entered, all jobs will run
if len(sys.argv) == 1:
    
    db_job_run_list = ['secure1', 'secure2', 'secure3', 'secure4', 'secure5', 'staging1', 'staging4', 'eu1', 'eustaging1', 'qaintx']
    
elif len(sys.argv) == 2 and sys.argv[1] == 'secure':
    
    db_job_run_list = ['secure1', 'secure2', 'secure3', 'secure4', 'secure5']
        
elif len(sys.argv) == 2 and sys.argv[1] == 'staging':
    
    db_job_run_list = ['staging1', 'staging4']

elif len(sys.argv) == 2 and sys.argv[1] == 'eu':
    
    db_job_run_list = ['eu1', 'eustaging1']

else:
    
    #get all command line parameters beyond the script name
    db_job_run_list = sys.argv[1:]

    
#job API call parameters

auth_json = {"Authorization": f"Bearer {DATABRICKS_TOKEN_STRING}",
             "Content-Type": "application/json"
            }

FINAL_URL = WORKSPACE_URL + "/api/2.1/jobs/run-now"

#run through all db names and get the job ID associated with each
for this_db_job in db_job_run_list:
    
    try:
        
        this_job_id = db_name_to_job_id_dict[this_db_job]

        print("Running job for " + this_db_job + "..." + "\n")

        job_json = {"job_id": this_job_id,
                    "notebook_params":{
                        "job_source":"remote"
                    }
        }
    
        response = requests.post(FINAL_URL, json=job_json, headers=auth_json)
        print(response.json())
    
    except Exception:
        print("pod name " + this_db_job + " is not valid.")
