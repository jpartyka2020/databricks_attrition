import requests
import json

#use your value of DATABRICKS_TOKEN_STRING
DATABRICKS_TOKEN_STRING = "dapiad60e91111da52a2da2c6b03441f228c"

WORKSPACE_URL = "https://dbc-e12db167-30b0.cloud.databricks.com"

#attrition dev job ID
#JOB_ID = xxxxxxxxxxxxxxx

db_name_to_job_id_dict = {}

db_name_to_job_id_dict['dev'] = xxxxxxxxxxxxxxx

#for all prod databases
db_name_to_job_id_dict['secure1'] = xxxxxxxxxxxxxxx
db_name_to_job_id_dict['secure2'] = xxxxxxxxxxxxxxx
db_name_to_job_id_dict['secure3'] = xxxxxxxxxxxxxxx
db_name_to_job_id_dict['secure4'] = xxxxxxxxxxxxxxx
db_name_to_job_id_dict['secure5'] = xxxxxxxxxxxxxxx
db_name_to_job_id_dict['staging1'] = xxxxxxxxxxxxxxx
db_name_to_job_id_dict['staging4'] = xxxxxxxxxxxxxxx
db_name_to_job_id_dict['eu1'] = xxxxxxxxxxxxxxx
db_name_to_job_id_dict['eustaging1'] = xxxxxxxxxxxxxxx

#this is a list that will determine which jobs we run
db_job_run_list = []

db_job_run_list.append('dev')
db_job_run_list.append('secure1')
db_job_run_list.append('secure2')
db_job_run_list.append('secure3')
db_job_run_list.append('secure4')
db_job_run_list.append('secure5')
db_job_run_list.append('staging1')
db_job_run_list.append('staging4')
db_job_run_list.append('eu1')
db_job_run_list.append('eustaging1')


auth_json = {"Authorization": f"Bearer {DATABRICKS_TOKEN_STRING}",
             "Content-Type": "application/json"
            }

FINAL_URL = WORKSPACE_URL + "/api/2.1/jobs/run-now"

for this_db_job in db_job_run_list:
    
    this_job_id = db_name_to_job_id_dict[this_db_job]
    
    job_json = {"job_id": this_job_id,
            "notebook_params":{
                "job_source":"remote"
            }
        }
    
    response = requests.post(FINAL_URL, json=job_json, headers=auth_json)
    print(response.json())
