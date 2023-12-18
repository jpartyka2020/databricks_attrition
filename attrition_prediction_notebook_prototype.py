# Databricks notebook source
""" Reads in the created Model and Data Object associated and predicts. """
import pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import sklearn
import sys
import logging
import warnings
import os

 
import shap
import sklearn as sk

from datetime import datetime
from pickle import load, dump
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, roc_auc_score, make_scorer
from sklearn.utils import resample
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.impute import IterativeImputer, SimpleImputer
from platform import python_version

from pyspark.sql import Row

#disable logging for now
logging.disable(logging.CRITICAL)


# COMMAND ----------

warnings.filterwarnings("ignore") 

print("python version:", python_version())
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("pickle version:", pickle.format_version)
print("sklearn version:", sklearn.__version__)

# COMMAND ----------

#constants to control how this notebook is run
TEST_MODE_ON = True
USE_IMP_TRAIN_TEST_DATASET = True
USE_MATT_MODEL = False
MAX_COL = 56

# COMMAND ----------

#we don't need to supply command line parameters in the Databricks environment.

gregorian_header_path_cl = ''
gregorian_file_path_cl = '/dbfs/FileStore/attrition_cleaned_data_files/cleaned_gregorian_data.tsv'
fiscal_header_path_cl = ''
fiscal_file_path_cl = '/dbfs/FileStore/attrition_cleaned_data_files/cleaned_fiscal_data.tsv'
gregorian_output_path = '/dbfs/FileStore/prediction_output/'
gregorian_vis_output_path = '/dbfs/FileStore/visual_output/'
fiscal_output_path = '/dbfs/FileStore/prediction_output/'
fortnightly_prediction_output_path = '/dbfs/FileStore/prediction_output/fortnightly_prediction_files'
fiscal_vis_output_path = '/dbfs/FileStore/visual_output/'
model_input_cl = '/dbfs/FileStore/pickleFiles'
fortnightly_log_file_path = '/dbfs/FileStore/prediction_output/fortnightly_log_files'

# COMMAND ----------

# Snowflake credentials
user = "app_datascience"
password = "Xactly123"

# COMMAND ----------

#Connect to Snowflake

options = {
  "sfUrl": "https://xactly-xactly_engg_datalake_aws.snowflakecomputing.com/",
  "sfUser": user,
  "sfPassword": password,
  "sfDatabase": "XTLY_ENGG",
  "sfSchema": "INSIGHTS",
  "sfWarehouse": "DIS_LOAD_WH",
  "truncate_table" : "ON",
  "usestagingtable" : "OFF",
}

# COMMAND ----------

""" 
Mimics the cleaning strategies used during Training of the model. Pickle objects of classes, this time a Cleaning Class,
need a local file to format off of in order to incorporate the components of the pickle file.
"""
class CleanData(object):
    def __init__(self):
        self.df = None
        self.num_one_hot = None
        self.one_hot_features = None
        self.tot_col = None
        self.var_df = None
        self.one_hot_df = None
        self.train = None
        self.test = None
        self.train_idx = None
        self.test_idx = None
        self.imp = None
        self.features = None
        self.dropped_columns = None
        self.norm_scaler = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None 

# COMMAND ----------

class LivePrediction(object):
    """ Creates a prediction object that can then be used to clean the data and output prediction. """

    def __init__(self, type_model, header_path, data_path, pred_output_path, visual_output_path, model_path, fortnightly_prediction_output_path, fortnightly_log_file_path, test_mode, options):
        """
        Initialized the LivePrediction Class.

        Keyword Arguments:
        type_model -- either 'fiscal' or 'gregorian'
        header_path -- the filepath to retrieve the headers of the data
        data_path -- the filepath to retreive data
        pred_output_path -- the filepath to output predictions
        visual_output_path -- the filepath to output visual weights
        model_path -- the filepath to retrieve the model
        fortnightly_prediction_output_path -- the path where you can find prediction output files
        fortnightly_log_file_path -- the path where you can find log files for a given model run
        """
        prefix_str = ""   #change to empty string for local run.

        self.type_model = type_model
        self.data_path = data_path
        self.pred_output_path = prefix_str + pred_output_path
        self.visual_output_path = prefix_str + visual_output_path
        self.model_path = model_path
        self.fortnightly_prediction_output_path = fortnightly_prediction_output_path
        self.fortnightly_log_file_path = fortnightly_log_file_path
        self.test_mode = test_mode
        self.options = options
        self.df = None

        #we will log events using a string
        self.log_file_str = ""
        self.log_file_str += "Start attrition model run: " + datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '\n'
        self.log_file_str += "====================" + "\n"

        if self.type_model == 'fiscal':

            if TEST_MODE_ON == True:
                self.df = pd.read_csv('/dbfs/FileStore/attrition_test_data/fiscal_input_prob_match_test.tsv', sep='\t')
            else:
                #read attrition data from Snowflake
                self.df = spark.read \
                        .format("snowflake") \
                        .options(**self.options) \
                        .option("dbtable","MERGE_FEATURE_FISCAL") \
                        .load().toPandas()
                
                #print(self.df.columns.tolist())                
                #print("shape of Snowflake dataset: " + str(self.df.shape))
                #sys.exit(0)
            
            print('Loaded Fiscal Data')
            self.log_file_str += "Just loaded gregorian data successfully" + "\n"

        
        elif self.type_model == 'gregorian':

            if TEST_MODE_ON == True:
                self.df = pd.read_csv('/dbfs/FileStore/attrition_test_data/gregorian_attrition_1K.tsv', sep='\t')
            else:
                self.df = spark.read \
                        .format("snowflake") \
                        .options(**self.options) \
                        .option("dbtable","MERGE_FEATURE_GREGORIAN") \
                        .load().toPandas()
                
                #print(self.df.columns.tolist())
                #print("shape of Snowflake dataset: " + str(self.df.shape))
                #sys.exit(0)

            print('Loaded Gregorian Data')
            self.log_file_str += "Just loaded gregorian data successfully" + "\n"

        else:
            self.log_file_str += "Could not load data" + "\n"
            raise NameError('HiThere')

        dtypes_series = self.df.dtypes

        dtypes_dict = dtypes_series.apply(lambda x: x.name).to_dict()
        self.df = self.df.astype(dtypes_dict)

        self.df['BUSINESS_ID'] = self.df['BUSINESS_ID'].astype('int')

        #remove TERM_AS_OF_DATE and TERM_NEXT_YEAR cols
        if self.df.shape[1] == MAX_COL:
            self.df = self.df.iloc[:, 0:-2]
        
        try:
            self.df.reset_index(
                drop=True, inplace=True
            )

            self.log_file_str += 'Array length and header length match' + '\n'

        except AssertionError as a:
            msg = "Array length and header length mismatch."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(a)
            raise AssertionError(msg)

        try:
            if self.type_model == 'fiscal':

                self.log_file_str += 'self.type_model = fiscal' + '\n'

                self.data_object = load(
                    open(self.model_path + '/data_object_fiscal.obj', 'rb')
                )

                self.log_file_str += 'data_object_fiscal.obj successfully loaded' + '\n'

                self.model = load(
                    open(self.model_path + '/rf_model_fiscal_matt.pkl', 'rb')
                )


                if 'CAL_YEAR' in self.df.columns:
                    self.df.rename(
                        columns={"CAL_YEAR": "FISCAL_YEAR"}, inplace=True
                    )

                self.log_file_str += 'CAL_YEAR renamed to FISCAL_YEAR' + '\n'

            elif self.type_model == 'gregorian':

                self.log_file_str += 'self.type_model = gregorian' + '\n'

                self.data_object = load(
                    open(self.model_path + '/data_object_gregorian.obj', 'rb')
                )

                self.log_file_str += 'data_object_gregorian.obj successfully loaded' + '\n'

                self.model = load(
                    open(self.model_path + '/rf_model_gregorian_matt.pkl', 'rb')
                )

                self.log_file_str += 'rf_model_gregorian.pkl successfully loaded' + '\n'

            else:
                msg = "Incorrect model flag hardcoded in script."
                self.log_file_str += msg
                self.write_log_file()
                #logger.critical(msg)
                raise ValueError(msg)
        except FileNotFoundError as f:
            msg = "Pickle files could not be found - either path is wrong or name of pickle files."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(f)
            raise FileNotFoundError(msg)
        
        self.business_ids = None
        self.master_participant_id = None
        self.fiscal_year = None
        self.cal_year = None
        self.term_as_of_date = None
        self.results = None
    
    def write_log_file(self):

        #log file name will have a datetime
        current_datetime_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

        log_file_name_str = self.fortnightly_log_file_path + '/log_file_' + self.type_model + '_' + current_datetime_str + '.tsv'

        with open(log_file_name_str, 'w') as writefile:
            writefile.write(self.log_file_str)


    def clean_input(self):
        """ Cleans the input in the same manner as the training of the model. """

        #Drop unused features, save to object BUSINESS_ID, MASTER_PARTICIPANT_ID:
        try:

            self.df['BUSINESS_ID'] = self.df['BUSINESS_ID'].astype(int)
            self.df['MASTER_PARTICIPANT_ID'] = self.df['MASTER_PARTICIPANT_ID'].astype(int)

            self.business_ids = self.df['BUSINESS_ID']
            self.master_participant_id = self.df['MASTER_PARTICIPANT_ID']
            
            if self.type_model == 'fiscal':

                self.df['FISCAL_YEAR'] = self.df['FISCAL_YEAR'].astype(int)
                self.fiscal_year = self.df['FISCAL_YEAR']

                self.df.drop(
                    ['BUSINESS_ID', 'HOME_CITY', 'HOME_STATE_PROVINCE', 'HOME_COUNTRY_CODE', 
                    'HIRE_AS_OF_DATE', 'TITLE_NAME_YR_END', 'FISCAL_YEAR', 'MASTER_PARTICIPANT_ID'],
                    axis=1, inplace=True
                )
            
            elif self.type_model == 'gregorian':
                self.df['CAL_YEAR'] = self.df['CAL_YEAR'].astype(int)
                self.cal_year = self.df['CAL_YEAR']
                self.df.drop(
                    ['BUSINESS_ID', 'HOME_CITY', 'HOME_STATE_PROVINCE', 'HOME_COUNTRY_CODE',
                    'HIRE_AS_OF_DATE', 'TITLE_NAME_YR_END', 'CAL_YEAR', 'MASTER_PARTICIPANT_ID'],
                    axis=1, inplace=True
                )

            self.df.drop(
                self.data_object.dropped_columns,
                axis=1, inplace=True
            )

            #Map Categorical Variables:
            self.df['TITLE_CATEGORY_YR_END'] = self.df['TITLE_CATEGORY_YR_END'].map(
                {'ACCOUNT_EXECUTIVE': 'ACCOUNT_EXECUTIVE', 'ACCOUNT_MANAGER': 'ACCOUNT_EXECUTIVE',
                'FIELD_SALES_REPRESENTATIVE': 'REPRESENTATIVE', 'INSIDE_SALES_REPRESENTATIVE': 'REPRESENTATIVE', 'LEAD_GENERATION_REPRESENTATIVE': 'REPRESENTATIVE',
                'BUSINESS_DEVELOPMENT_ALLIANCE_PARTNERSHIP_REP': 'REPRESENTATIVE', 'SERVICES_REPRESENTATIVE': 'REPRESENTATIVE',
                'CUSTOMER_SUCCESS_REPRESENTATIVE': 'REPRESENTATIVE', 'CHANNEL_SALES_REPRESENTATIVE': 'REPRESENTATIVE',
                'SALES_SYSTEMS_APPLICATIONS_ENGINEER': 'CONSULTANT', 'SALES_SUPPORT_OPERATIONS': 'CONSULTANT', 'PRE_SALES_CONSULTANT': 'CONSULTANT',
                'PRODUCT_INDUSTRY_SPECIALIST': 'CONSULTANT', 'SALES_MANAGER': 'MANAGER', 'SERVICES_MANAGER': 'MANAGER', 'STRATEGIC_KEY_GLOBAL_ACCOUNT_MANAGER': 'MANAGER',
                'MANAGER_SALES_SYSTEMS_APPLICATION_ENGINEERING': 'MANAGER', 'MANAGER_CHANNEL_SALES': 'MANAGER', 'MANAGER_BUSINESS_DEVELOPMENT_ALLIANCE_PARTNERSHIPS': 'MANAGER',
                'MANAGER_INSIDE_SALES': 'MANAGER', 'MANAGER_LEAD_GENERATION': 'MANAGER', 'SALES_DIRECTOR': 'DIRECTOR', 'SALES_EXECUTIVE': 'DIRECTOR'
                }
            )

            self.df['INDUSTRY'] = self.df['INDUSTRY'].map(
                {'Software': 'SaaS & Cloud', 'Media & Internet': 'SaaS & Cloud', 'SaaS & Cloud': 'SaaS & Cloud',
                'Financial Services': 'Business Services', 'Business Services': 'Business Services',
                'Travel & Hospitality': 'Communications', 'Communications': 'Communications',
                'Medical Devices': 'Life Sciences & Pharma','Healthcare': 'Life Sciences & Pharma', 'Life Sciences & Pharma': 'Life Sciences & Pharma',
                'Consumer Goods': 'Retail', 'Retail': 'Retail',
                'Manufacturing': 'Manufacturing', 'High-Tech Manufacturing': 'Manufacturing',
                'Energy': 'Government/Public Sector', 'Other': 'Government/Public Sector', 'Government/Public Sector': 'Government/Public Sector'
                }
            )
            self.df['OWNERSHIP'] = self.df['OWNERSHIP']         #Checking for key error on OWNERSHIP

        except KeyError as k:
            msg = "The Data is missing some column that was present in training."

            print(self.df.columns.tolist())

            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(k)
            raise KeyError(msg)


        #Add Difference Variables:

        #set data types diff-related columns
        self.df['COUNT_MAX_MONTH_ORDER'] = self.df['COUNT_MAX_MONTH_ORDER'].astype(float)
        self.df['COUNT_MIN_MONTH_ORDER'] = self.df['COUNT_MIN_MONTH_ORDER'].astype(float)
        self.df['MAX_QUOTA_AMT_USD'] = self.df['MAX_QUOTA_AMT_USD'].astype(float)
        self.df['MIN_QUOTA_AMT_USD'] = self.df['MIN_QUOTA_AMT_USD'].astype(float)

        try:

            self.df['DIFF_MONTH_ORDER'] = (self.df['COUNT_MAX_MONTH_ORDER']
                                        - self.df['COUNT_MIN_MONTH_ORDER'])

            self.df['DIFF_QUOTA_AMT_USD'] = (self.df['MAX_QUOTA_AMT_USD']
                                            - self.df['MIN_QUOTA_AMT_USD'])
        
        except TypeError as t:
            msg = "Cannot do arithmetic operations on strings."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(t)
            raise TypeError(msg)
            

        #One Hot Encode Categorical Columns:
        num_one_hot = (len(np.unique(self.df['TITLE_CATEGORY_YR_END'].dropna()))
                    + len(np.unique(self.df['INDUSTRY'].dropna()))
                    + len(np.unique(self.df['OWNERSHIP'].dropna())))

        for feature in ['TITLE_CATEGORY_YR_END', 'INDUSTRY', 'OWNERSHIP']:
            
            self.df = pd.concat(
                [self.df, pd.get_dummies(self.df[feature], prefix=feature)], axis=1
            )
            self.df.drop(
                [feature], axis=1,
                inplace=True
            )

        #Impute/Normalize Data - Only imputes on numeric columns:
        tot_col = self.df.shape[1]
        one_hot_df = self.df.iloc[:, (tot_col-num_one_hot):(tot_col)]
        one_hot_features = one_hot_df.columns
        var_df = self.df.iloc[:, 0:(tot_col-num_one_hot)]

        var_df_columns = var_df.columns.tolist()

        try:

            var_array = self.data_object.imp.transform(
                var_df.values
            )
            var_array = self.data_object.norm_scaler.transform(
                var_array
            )        
       
        except ValueError as v:
            msg = "Array passed into imputer or normalizer are incorrect lengths."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(v)
            raise ValueError(msg)
        

        var_df = pd.DataFrame(
            var_array, columns=var_df.columns
        )
        
        self.df = pd.concat(
            [var_df, one_hot_df], axis=1
        )

        feature_list = [
        'PR_TARGET_USD','SALARY_USD','COUNT_MONTH_EMPLOYED_TIL_DEC',
        'COUNT_MONTHS_GOT_PAYMENT','LAST_PAYMENT_UNTIL_YR_END','YEAR_PAYMENT',
        'COUNT_UNIQ_QUOTA','COUNT_AVG_MONTH_PAID_QUOTA',
        'LAST_QUOTA_PAID_UNTIL_YR_END','SUM_CREDIT_AMT_USD','MIN_CREDIT_AMT_USD',
        'DIFF_QUOTA_AMT_USD','COUNT_UNIQ_TITLE_NAME','COUNT_UNIQ_MGR_ID','INDUSTRY_SaaS & Cloud',
        'TITLE_CATEGORY_YR_END_ACCOUNT_EXECUTIVE',
        'TITLE_CATEGORY_YR_END_CONSULTANT','TITLE_CATEGORY_YR_END_DIRECTOR',
        'TITLE_CATEGORY_YR_END_MANAGER','TITLE_CATEGORY_YR_END_REPRESENTATIVE','OWNERSHIP_Private','OWNERSHIP_Public',
        'INDUSTRY_Business Services','INDUSTRY_Communications','INDUSTRY_Government/Public Sector','INDUSTRY_Life Sciences & Pharma',
        'INDUSTRY_Manufacturing']

        if self.type_model == 'gregorian':
            feature_list += ['COMP_AVG_MONTH_PAYEECOUNT']

        #Ensure all Categorical Features Present:
        try:

            #for col in self.data_object.one_hot_df.columns:
            for col in feature_list:
                if col not in one_hot_features:
                    self.df[col] = 0

        except AttributeError as a:
            msg = "Incompatible versions of Pandas between Pickle File and VM."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(a)
            raise AttributeError(msg)

        #set data types for all other columns in self.df
        self.df['PR_TARGET_USD'] = self.df['PR_TARGET_USD'].astype(float)
        self.df['SALARY_USD'] = self.df['SALARY_USD'].astype(float)
        self.df['COUNT_MONTH_EMPLOYED_TIL_DEC'] = self.df['COUNT_MONTH_EMPLOYED_TIL_DEC'].astype(float)
        self.df['COUNT_MONTHS_GOT_PAYMENT'] = self.df['COUNT_MONTHS_GOT_PAYMENT'].astype(float)
        self.df['LAST_PAYMENT_UNTIL_YR_END'] = self.df['LAST_PAYMENT_UNTIL_YR_END'].astype(float)
        self.df['YEAR_PAYMENT'] = self.df['YEAR_PAYMENT'].astype(float)
        self.df['COUNT_UNIQ_QUOTA'] = self.df['COUNT_UNIQ_QUOTA'].astype(float)
        self.df['COUNT_AVG_MONTH_PAID_QUOTA'] = self.df['COUNT_AVG_MONTH_PAID_QUOTA'].astype(float)
        self.df['LAST_QUOTA_PAID_UNTIL_YR_END'] = self.df['LAST_QUOTA_PAID_UNTIL_YR_END'].astype(float)
        self.df['SUM_CREDIT_AMT_USD'] = self.df['SUM_CREDIT_AMT_USD'].astype(float)
        self.df['MIN_CREDIT_AMT_USD'] = self.df['MIN_CREDIT_AMT_USD'].astype(float)
        self.df['COUNT_UNIQ_TITLE_NAME'] = self.df['COUNT_UNIQ_TITLE_NAME'].astype(float)
        self.df['COUNT_UNIQ_MGR_ID'] = self.df['COUNT_UNIQ_MGR_ID'].astype(float)
        self.df['TITLE_CATEGORY_YR_END_ACCOUNT_EXECUTIVE'] = self.df['TITLE_CATEGORY_YR_END_ACCOUNT_EXECUTIVE'].astype(int)
        self.df['TITLE_CATEGORY_YR_END_CONSULTANT'] = self.df['TITLE_CATEGORY_YR_END_CONSULTANT'].astype(int)
        self.df['TITLE_CATEGORY_YR_END_DIRECTOR'] = self.df['TITLE_CATEGORY_YR_END_DIRECTOR'].astype(int)
        self.df['TITLE_CATEGORY_YR_END_MANAGER'] = self.df['TITLE_CATEGORY_YR_END_MANAGER'].astype(int)
        self.df['TITLE_CATEGORY_YR_END_REPRESENTATIVE'] = self.df['TITLE_CATEGORY_YR_END_REPRESENTATIVE'].astype(int)
        self.df['INDUSTRY_SaaS & Cloud'] = self.df['INDUSTRY_SaaS & Cloud'].astype(int)
        self.df['OWNERSHIP_Private'] = self.df['OWNERSHIP_Private'].astype(int)
        self.df['OWNERSHIP_Public'] = self.df['OWNERSHIP_Public'].astype(int)
        self.df['INDUSTRY_Business Services'] = self.df['INDUSTRY_Business Services'].astype(int)
        self.df['INDUSTRY_Communications'] = self.df['INDUSTRY_Communications'].astype(int)
        self.df['INDUSTRY_Government/Public Sector'] = self.df['INDUSTRY_Government/Public Sector'].astype(int)
        self.df['INDUSTRY_Life Sciences & Pharma'] = self.df['INDUSTRY_Life Sciences & Pharma'].astype(int)
        self.df['INDUSTRY_Manufacturing'] = self.df['INDUSTRY_Manufacturing'].astype(int)

        if self.type_model == 'gregorian':
            self.df['COMP_AVG_MONTH_PAYEECOUNT'] = self.df['COMP_AVG_MONTH_PAYEECOUNT'].astype(int)
            

        #print("Features from self.data_object.features = " + str(self.data_object.features))
        #print('---------')
        #print("Features from self.df = " + str(self.df.columns.tolist()))

        self.df = self.df[feature_list]
        #self.df = self.df[self.data_object.features]

        print("bottom clean_input")
        print("Cleaned Data Successfully.")

    def get_external_train_test_data(self):
        
        #load outside train_test data
        df_external_data = None

        if self.type_model == "fiscal":
            df_external_data = pd.read_csv('/dbfs/FileStore/attrition_test_data/fiscal_attrition.tsv', sep='\t')
        
        if self.type_model == "gregorian":
            df_external_data = pd.read_csv('/dbfs/FileStore/attrition_test_data/gregorian_attrition.tsv', sep='\t')
        
        #create one-hot-encoded variables for Industry and Ownership
        df_industry = pd.get_dummies(df_external_data['INDUSTRY'], prefix='INDUSTRY')
        df_ownership = pd.get_dummies(df_external_data['OWNERSHIP'], prefix='OWNERSHIP')

        model_feature_list = ['PR_TARGET_USD', 'SALARY_USD', 'COUNT_MONTH_EMPLOYED_TIL_DEC', 'COUNT_MONTHS_GOT_PAYMENT', 'LAST_PAYMENT_UNTIL_YR_END', 'YEAR_PAYMENT', 'COUNT_UNIQ_QUOTA', 'COUNT_AVG_MONTH_PAID_QUOTA', 'LAST_QUOTA_PAID_UNTIL_YR_END', 'SUM_CREDIT_AMT_USD', 'MIN_CREDIT_AMT_USD', 'DIFF_QUOTA_AMT_USD', 'COUNT_UNIQ_TITLE_NAME', 'COUNT_UNIQ_MGR_ID','OWNERSHIP_Public', 'OWNERSHIP_Private', 'TERM_NEXT_YEAR','TITLE_CATEGORY_YR_END_ACCOUNT_EXECUTIVE', 'TITLE_CATEGORY_YR_END_CONSULTANT', 'TITLE_CATEGORY_YR_END_DIRECTOR', 'TITLE_CATEGORY_YR_END_MANAGER', 'TITLE_CATEGORY_YR_END_REPRESENTATIVE', 'INDUSTRY_SaaS & Cloud', 'INDUSTRY_Business Services', 'INDUSTRY_Communications', 'INDUSTRY_Government/Public Sector', 'INDUSTRY_Life Sciences & Pharma', 'INDUSTRY_Manufacturing']

        if self.type_model == "gregorian":
            model_feature_list += ['COMP_AVG_MONTH_PAYEECOUNT']

        #ownership_column_list = ['OWNERSHIP_Public','OWNERSHIP_Private']

        #drop industry and ownership columns in df_external_data
        df_external_data = df_external_data.drop(['INDUSTRY','OWNERSHIP'], axis=1)

        #map values in TITLE_CATEGORY_YR_END column of df_external_data
        df_external_data['TITLE_CATEGORY_YR_END'] = df_external_data['TITLE_CATEGORY_YR_END'].map(
                    {'ACCOUNT_EXECUTIVE': 'ACCOUNT_EXECUTIVE', 'ACCOUNT_MANAGER': 'ACCOUNT_EXECUTIVE',
                    'FIELD_SALES_REPRESENTATIVE': 'REPRESENTATIVE', 'INSIDE_SALES_REPRESENTATIVE': 'REPRESENTATIVE', 'LEAD_GENERATION_REPRESENTATIVE': 'REPRESENTATIVE',
                    'BUSINESS_DEVELOPMENT_ALLIANCE_PARTNERSHIP_REP': 'REPRESENTATIVE', 'SERVICES_REPRESENTATIVE': 'REPRESENTATIVE',
                    'CUSTOMER_SUCCESS_REPRESENTATIVE': 'REPRESENTATIVE', 'CHANNEL_SALES_REPRESENTATIVE': 'REPRESENTATIVE',
                    'SALES_SYSTEMS_APPLICATIONS_ENGINEER': 'CONSULTANT', 'SALES_SUPPORT_OPERATIONS': 'CONSULTANT', 'PRE_SALES_CONSULTANT': 'CONSULTANT',
                    'PRODUCT_INDUSTRY_SPECIALIST': 'CONSULTANT', 'SALES_MANAGER': 'MANAGER', 'SERVICES_MANAGER': 'MANAGER', 'STRATEGIC_KEY_GLOBAL_ACCOUNT_MANAGER': 'MANAGER',
                    'MANAGER_SALES_SYSTEMS_APPLICATION_ENGINEERING': 'MANAGER', 'MANAGER_CHANNEL_SALES': 'MANAGER', 'MANAGER_BUSINESS_DEVELOPMENT_ALLIANCE_PARTNERSHIPS': 'MANAGER',
                    'MANAGER_INSIDE_SALES': 'MANAGER', 'MANAGER_LEAD_GENERATION': 'MANAGER', 'SALES_DIRECTOR': 'DIRECTOR', 'SALES_EXECUTIVE': 'DIRECTOR'
                    }
                )
        
        title_category_yr_end_column_list = ['TITLE_CATEGORY_YR_END_ACCOUNT_EXECUTIVE', 'TITLE_CATEGORY_YR_END_CONSULTANT', 'TITLE_CATEGORY_YR_END_DIRECTOR', 'TITLE_CATEGORY_YR_END_MANAGER', 'TITLE_CATEGORY_YR_END_REPRESENTATIVE']

        df_title_category_yr_end = pd.get_dummies(df_external_data['TITLE_CATEGORY_YR_END'], prefix='TITLE_CATEGORY_YR_END')
        
        df_external_data = df_external_data.drop(['TITLE_CATEGORY_YR_END'], axis=1)

        #add any OHE columns to df_external_data
        


        #create DIFF_QUOTA_AMT_USD within df_external_data
        df_external_data['DIFF_QUOTA_AMT_USD'] = df_external_data['MAX_QUOTA_AMT_USD'] - df_external_data['MIN_QUOTA_AMT_USD']
        df_external_data = pd.concat([df_external_data, df_industry, df_ownership, df_title_category_yr_end], axis=1)

        #filter out reporting columns
        reporting_columns_list = ['BUSINESS_ID', 'CAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID']
        index_nonreporting_columns = df_external_data.columns.difference(reporting_columns_list)
        df_external_data = df_external_data[index_nonreporting_columns]

        #Ensure all Categorical Features Present:
        try:

            df_ed_col_list = df_external_data.columns.tolist()

            #for col in self.data_object.one_hot_df.columns:
            for col in model_feature_list:
                if col not in df_ed_col_list:
                    df_external_data[col] = 0

        except AttributeError as a:
            msg = "Incompatible versions of Pandas between Pickle File and VM."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(a)
            raise AttributeError(msg)

        df_external_data = df_external_data[model_feature_list]

        #fill in missing values instead of imputing
        df_external_data.fillna(df_external_data.mean(), inplace=True)
        
        #split df_external_data to train and test
        [df_external_train, df_external_test] = train_test_split(df_external_data, test_size = 1/3, random_state = 42)

        return [df_external_train, df_external_test]


    def predict(self):

        self.log_file_str += '\n' + 'In predict()' + '\n'
        self.log_file_str += '====================' + '\n'

        if USE_IMP_TRAIN_TEST_DATASET == True:
            [df_external_train, df_external_test] = self.get_external_train_test_data()

            print(df_external_train.columns.tolist())
            print('--------')
            print(df_external_test.columns.tolist())

            [df_train, df_test] = train_test_split(self.df, test_size = 1/3, random_state = 42)

            df_external_train_target = df_external_train[['TERM_NEXT_YEAR']]
            df_external_train = df_external_train.drop(['TERM_NEXT_YEAR'], axis=1)

            df_external_test_target = df_external_test[['TERM_NEXT_YEAR']]
            df_external_test = df_external_test.drop(['TERM_NEXT_YEAR'], axis=1)
        else:
            pass
            #[df_train, df_test] = train_test_split(self.df, test_size = 1/3, random_state = 42)

        #df_train_target = df_train[['TERM_NEXT_YEAR']]
        #df_train = df_train.drop(['TERM_NEXT_YEAR'], axis=1)

        #df_test_target = df_test[['TERM_NEXT_YEAR']]
        #df_test = df_test.drop(['TERM_NEXT_YEAR'], axis=1)

        y_test_pred = None
        train_score = None
        test_score = None

        if USE_IMP_TRAIN_TEST_DATASET == True:

            x_external_train = df_external_train.values
            y_external_train = df_external_train_target.values

            x_external_test = df_external_test.values
            y_external_test = df_external_test_target.values

            rf_max_features = len(df_train.columns)

            if USE_MATT_MODEL == False:
                self.model = RandomForestClassifier(n_estimators = 525, criterion = 'entropy', max_depth = 100, max_features = 'log2', min_samples_split = 5, ccp_alpha=0.000001, random_state=42)

                self.model.fit(x_external_train, y_external_train)
                y_test_pred = self.model.predict(x_external_test)

                train_score = self.model.score(x_external_train, y_external_train)
                test_score = self.model.score(x_external_test, y_external_test)

                print("Test score: ", test_score, "Train score: ", train_score)
                self.log_file_str += 'Train score: ' + str(train_score) + '\n'
                self.log_file_str += 'Test score: ' + str(test_score) + '\n'
        
        self.log_file_str += 'Successfully trained RandomForestClassifier' + '\n'

        try:
            probs = self.model.predict_proba(self.df.values)
        except ValueError as v:
            msg = "Array passed into model is of incorrect length."
            #logger.critical(msg)
            #logger.critical(v)
            raise ValueError(msg)
        except:
            msg = "Unspecified Error when making prediction."
            #logger.critical(msg)
            raise ValueError(msg)


        if self.type_model == 'fiscal':
            
            d = {
                'BUSINESS_ID': self.business_ids, 'MASTER_PARTICIPANT_ID': self.master_participant_id,
                'FISCAL_YEAR': self.fiscal_year, 'PRED_TERM_PROB': probs[:, 1]
                }
            
            try:

                self.results = pd.DataFrame(d)

                if self.test_mode == True:

                    self.results.to_csv(self.pred_output_path + "results_test_fiscal.csv", index=False)
                    self.log_file_str += 'Generated results_test_fiscal.csv successfully: ' + '\n'

                else:

                    df_spark = spark.createDataFrame(self.results)

                    df_spark.write \
                        .format("snowflake") \
                        .mode("append") \
                        .options(**options) \
                        .option("dbtable", "TURNOVER_PRED_FISCAL") \
                        .save()
                    
                    self.log_file_str += 'Wrote fiscal self.results to Snowflake successfully: ' + '\n'
                
                #get today's date and time as a string
                current_datetime_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

                #write data to Databricks as well, whether we are using a test file or not
                self.results.to_csv(
                    self.fortnightly_prediction_output_path + '/part_fiscal_pred_' + current_datetime_str + '.tsv',
                    index=False, sep='\t'
                )

                self.log_file_str += 'Wrote fiscal fortnightly_prediction file successfully: ' + '\n'


            except FileNotFoundError as f:
                msg = "Fiscal output filepath not found when outputting."
                self.log_file_str += msg
                self.write_log_file()
                #logger.critical(msg)
                #logger.critical(f)
                raise FileNotFoundError(msg)

        elif self.type_model == 'gregorian':

            d = {
                'BUSINESS_ID': self.business_ids, 'MASTER_PARTICIPANT_ID': self.master_participant_id,
                'CAL_YEAR': self.cal_year, 'PRED_TERM_PROB': probs[:, 1]
                }
            
            try:

                self.results = pd.DataFrame(d)

                if self.test_mode == True:

                    self.results.to_csv(self.pred_output_path + "results_test_gregorian.csv", index=False)
                    self.log_file_str += 'Generated results_test_gregorian.csv successfully: ' + '\n'

                else:

                    df_spark = spark.createDataFrame(self.results)

                    df_spark.write \
                        .format("snowflake") \
                        .mode("append") \
                        .options(**options) \
                        .option("dbtable", "TURNOVER_PRED_GREGORIAN") \
                        .save()
                
                self.log_file_str += 'Wrote gregorian self.results to Snowflake successfully: ' + '\n'
            
                #get today's date and time as a string
                current_datetime_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

                #write data to Databricks as well, whether we are using a test file or not
                self.results.to_csv(
                    self.fortnightly_prediction_output_path + '/part_gregorian_pred_' + current_datetime_str + '.tsv',
                    index=False, sep='\t'
                )

                self.log_file_str += 'Wrote gregorian fortnightly_prediction file successfully: ' + '\n'


            except FileNotFoundError as f:
                msg = "Gregorian output filepath not found when outputting."
                self.log_file_str += msg
                self.write_log_file()
                #logger.critical(msg)
                #logger.critical(f)
                raise FileNotFoundError(msg)

        print("Predictions outputted successfully.")
        print("SUCCESS.")
        self.log_file_str += 'Predictions outputted successfully' + '\n'
        self.log_file_str += 'SUCCESS' + '\n'

    def explain_pred(self):
        """ 
        Explain the predictions for the input data, outputted as WEIGHTS_ATTRITION.tsv 
        DO NOT RUN - Need to import SHAP
        """

        print("Explainer started.")
        self.log_file_str += '\n' + 'In explain_pred()' + '\n'
        self.log_file_str += '====================' + '\n'

        shap_exp = shap.TreeExplainer(
            self.model, background=self.data_object.x_train
        )

        print("Shap Explainer Created.")
        self.log_file_str += 'Shap Explainer Created' + '\n'

        shap_values_test = shap_exp.shap_values(
            self.df.values, approximate=True,
            check_additivity=False
        )

        print("Shap Values Created.")
        self.log_file_str += 'Shap Values Created' + '\n'

        shap_arr = np.zeros(
            self.df.shape
        )

        try:
            for row in range(len(self.df)):
                vals = shap_values_test[1][row]
                shap_arr[row] = vals
            
            self.log_file_str += 'shap_arr created' + '\n'
            print("shap_arr created")

        except IndexError as i:
            msg = "No shap values were created from the TreeExplainer."
            self.log_file_str += msg
            self.write_log_file()
            raise IndexError(msg)

        shap_results = pd.DataFrame(
            shap_arr, columns=self.df.columns
        )
        self.results = pd.concat(
            [self.results, shap_results], axis=1
        )

        print("self.results created")

        #Combine Shap Values of One-Hot-Encoded Features:
        for prefix in ['TITLE_CATEGORY_YR_END', 'INDUSTRY', 'OWNERSHIP']:
            filter_col = [col for col in self.results.columns if col.startswith(prefix)]
            self.results[prefix] = self.results[filter_col].sum(axis=1)
            self.results.drop(
                filter_col,
                axis=1, inplace=True
            )
        self.results["EXPECTED_VALUE"] = shap_exp.expected_value[1]

        #if model is gregorian, drop COMP_AVG_MONTH_PAYEECOUNT
        if self.type_model == 'gregorian':
            self.results = self.results.drop(['COMP_AVG_MONTH_PAYEECOUNT'], axis=1)

        self.log_file_str += 'combined shap values with one-hot-encoded features successfully' + '\n'
        print("combined shap values with one-hot-encoded features successfully")

        #get today's date and time as a string
        current_datetime_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

        self.results.to_csv(
            self.visual_output_path + '/part_visual_' + self.type_model + "_" + current_datetime_str +  '.tsv',
            index=False, sep='\t'
        )

        self.log_file_str += 'wrote shap output file successfully' + '\n'
        print("wrote shap output file successfully")

        if self.test_mode == False:

            snowflake_table_name = ""

            if self.type_model == 'fiscal':
                snowflake_table_name = "VISUALISATION_FISCAL"
            else:
                snowflake_table_name = "VISUALISATION_GREGORIAN"
            
            df_spark = spark.createDataFrame(self.results)

            df_spark.write \
                .format("snowflake") \
                .mode("append") \
                .options(**self.options) \
                .option("dbtable",snowflake_table_name) \
                .save()

        print("Explainer ended.")
        self.log_file_str += 'Explainer ended' + '\n'

# COMMAND ----------

#read the TRIGGER_ATTRITION table
trigger_attrition_read_query = f"(select * from INSIGHTS_PARAMETER where name = 'TRIGGER_ATTRITION') AS subquery"

df_trigger_attrition = spark.read \
                    .format("snowflake") \
                    .options(**options) \
                    .option("dbtable",trigger_attrition_read_query) \
                    .load().toPandas()

trigger_attrition_flag = ""

#if the flag is False,then we don't execute attrition
if TEST_MODE_ON == False:
    
    try:
        trigger_attrition_flag = df_trigger_attrition.loc[0, 'VALUE']
    except Exception:
        trigger_attrition_flag = "TRUE"

    print("trigger_attrition_flag = " + str(trigger_attrition_flag))

#this will help determine if the job is triggered by the scheduler or remotely
parameter_dict = dict(dbutils.notebook.entry_point.getCurrentBindings())

job_executed_remotely = False

if 'job_source' in parameter_dict:
    job_executed_remotely = True

if trigger_attrition_flag == 'TRUE' or TEST_MODE_ON == True or job_executed_remotely == True:

    #first, truncate Snowflake tables
    truncate_success = dbutils.notebook.run("attrition_truncate_snowflake_tables", 60)

    if truncate_success != "success":
        print("Not able to truncate Snowflake tables...stopping execution")
        sys.exit(0)

    print("let's start making predictions....")
    print('----------')

    #execute both fiscal and gregorian attrition

    fiscal_args = ["fiscal", fiscal_header_path_cl, fiscal_file_path_cl, fiscal_output_path, fiscal_vis_output_path, model_input_cl, fortnightly_prediction_output_path, fortnightly_log_file_path, TEST_MODE_ON, options]

    gregorian_args = ["gregorian", gregorian_header_path_cl, gregorian_file_path_cl, gregorian_output_path, gregorian_vis_output_path, model_input_cl, fortnightly_prediction_output_path, fortnightly_log_file_path, TEST_MODE_ON, options]

    for input_lst in [gregorian_args, fiscal_args]: 
        batch_pred = LivePrediction(
            input_lst[0], input_lst[1], input_lst[2],
            input_lst[3], input_lst[4], input_lst[5], 
            input_lst[6], input_lst[7], input_lst[8],
            input_lst[9]
        )

        batch_pred.clean_input()
        batch_pred.predict()
        batch_pred.explain_pred()

        #write log file
        batch_pred.write_log_file()
    
    if TEST_MODE_ON == False:
        
        try: 
            #set trigger_attrition_flag to FALSE
            df_trigger_attrition.loc[0, 'VALUE'] = 'FALSE'

            #convert df_trigger_attrition to a spark dataframe
            df_spark_trigger_attrition = spark.createDataFrame(df_trigger_attrition)

            df_spark_trigger_attrition.write \
                .format("snowflake") \
                .mode("append") \
                .options(**options) \
                .option("dbtable", "INSIGHTS_PARAMETER") \
                .save()
            
            print("Finished writing to INSIGHTS_PARAMETER table")

        except Exception:
            print("error writing to INSIGHTS_PARAMETER table. Skipping...")

else:

    print("flag is set to FALSE, attrition will not execute")

# COMMAND ----------


