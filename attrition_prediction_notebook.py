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

MAX_COL = 56

 #save cols into mandatory_reporting_columns and optional_reporting_columns for proper file processing within CleanData
OPTIONAL_REPORTING_COLUMNS_LIST = ['HOME_CITY', 'HOME_STATE_PROVINCE', 'HOME_COUNTRY_CODE','HIRE_AS_OF_DATE', 'TITLE_NAME_YR_END']
MANDATORY_REPORTING_COLUMNS_GREGORIAN = ['BUSINESS_ID', 'CAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID']
MANDATORY_REPORTING_COLUMNS_FISCAL = ['BUSINESS_ID', 'FISCAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID']

# COMMAND ----------

#constants to control how this notebook is run
TEST_MODE_ON = False
test_fiscal_data_path = ''
test_gregorian_data_path = ''

# COMMAND ----------

#global log file strings for fiscal and gregorian cleaning and predictions
fiscal_log_file_str = ""
gregorian_log_file_str = ""

# COMMAND ----------

#we don't need to supply command line parameters in the Databricks environment
#we also might not need these -- hmmm...

gregorian_header_path_cl = ''
gregorian_file_path_cl = '/dbfs/FileStore/attrition_test_data/cleaned_gregorian_data.tsv'
fiscal_header_path_cl = ''
fiscal_file_path_cl = '/dbfs/FileStore/attrition_test_data/cleaned_fiscal_data.tsv'
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

# options = {
#   "sfUrl": "https://xactly-xactly_engg_datalake.snowflakecomputing.com/",
#   "sfUser": user,
#   "sfPassword": password,
#   "sfDatabase": "XTLY_ENGG",
#   "sfSchema": "INSIGHTS",
#   "sfWarehouse": "DIS_LOAD_WH"
# }

#Connect to Snowflake

options = {
  "sfUrl": "https://xactly-xactly_engg_datalake_aws.snowflakecomputing.com/",
  "sfUser": user,
  "sfPassword": password,
  "sfDatabase": "XTLY_ENGG",
  "sfSchema": "INSIGHTS",
  "sfWarehouse": "DIS_LOAD_WH"
}

# COMMAND ----------

class CleanData(object):

    def __init__(self, calendar_type, test_mode=False, log_file_str=None):
        self.calendar_type = calendar_type
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

        self.df_mandatory_reporting_columns = None 
        self.df_optional_reporting_columns = None
        self.mandatory_reporting_columns_list = []
        self.log_file_str = log_file_str
        self.major_train_columns_list = []
        self.major_test_columns_list = []

        self.init_log_file()
        self.init_reporting_columns()
        self.init_train_test_columns()
        self.load_data() # Reads in csv.
        self.map_categorical()
        self.drop_ids()
        self.drop_columns()
        self.one_hot_encode_categorical()
        self.set_dependent_last()
        self.seperate_categorical_numeric()
        self.outlier_removal()
        self.reset_index()
        self.split_data()
        self.add_difference_variables()
        self.impute_data()
        self.recombine_data()
        self.normalize_data()
        self.select_features()
    
    def init_log_file(self):
        
        self.log_file_str += "Starting cleaning of " + self.calendar_type + " data" + "\n"
        self.log_file_str += "=====================" + "\n"
    
    def init_reporting_columns(self):

        global MANDATORY_REPORTING_COLUMNS_GREGORIAN
        global MANDATORY_REPORTING_COLUMNS_FISCAL

        if self.calendar_type == "gregorian":
            self.mandatory_reporting_columns_list = MANDATORY_REPORTING_COLUMNS_GREGORIAN
        else:
            self.mandatory_reporting_columns_list = MANDATORY_REPORTING_COLUMNS_FISCAL
    

    def init_train_test_columns(self):

        if self.calendar_type == "gregorian":
            self.major_train_columns_list = ['BUSINESS_ID', 'CAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID']
            self.major_test_columns_list = ['BUSINESS_ID', 'CAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID']
        else:
            self.major_train_columns_list = ['BUSINESS_ID', 'FISCAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID']
            self.major_test_columns_list = ['BUSINESS_ID', 'FISCAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID']


    def load_data(self):
        """Read in the data:"""

        global TEST_MODE_ON

        int_col_list = ['MASTER_PARTICIPANT_ID','CAL_YEAR','HIRE_AS_OF_DATE','COUNT_UNIQ_TITLE_NAME','COUNT_UNIQ_MGR_ID', 'COUNT_MONTHS_GOT_PAYMENT', 'TERM_NEXT_YEAR']

        obj_col_list = ['TITLE_NAME_YR_END','TITLE_CATEGORY_YR_END','HOME_CITY','HOME_STATE_PROVINCE','HOME_COUNTRY_CODE',
                        'INDUSTRY','OWNERSHIP']

        if self.calendar_type == 'fiscal':

            if TEST_MODE_ON == True:
                self.df = pd.read_csv('/dbfs/FileStore/attrition_test_data/fiscal_attrition.tsv', sep='\t')
            else:
                #read attrition data from Snowflake
                self.df = spark.read \
                        .format("snowflake") \
                        .options(**options) \
                        .option("dbtable","MERGE_FEATURE_FISCAL") \
                        .load().toPandas()
                
                #print("shape of Snowflake dataset: " + str(self.df.shape))
                #sys.exit(0)
            
            dtypes_series = self.df.dtypes

            dtypes_dict = dtypes_series.apply(lambda x: x.name).to_dict()
            self.df = self.df.astype(dtypes_dict)

            self.df['BUSINESS_ID'] = self.df['BUSINESS_ID'].astype('int')           
            
            print('Loaded Fiscal Data')
            self.log_file_str += "Just loaded data successfully" + "\n"
        
        elif self.calendar_type == 'gregorian':

            if TEST_MODE_ON == True:
                self.df = pd.read_csv('/dbfs/FileStore/attrition_test_data/gregorian_attrition.tsv', sep='\t')
            else:
                self.df = spark.read \
                        .format("snowflake") \
                        .options(**options) \
                        .option("dbtable","MERGE_FEATURE_GREGORIAN") \
                        .load().toPandas()
                
                #print("shape of Snowflake dataset: " + str(self.df.shape))
                #sys.exit(0)
                

            dtypes_series = self.df.dtypes

            dtypes_dict = dtypes_series.apply(lambda x: x.name).to_dict()
            self.df = self.df.astype(dtypes_dict)

            print('Loaded Gregorian Data')
            self.log_file_str += "Just loaded data successfully" + "\n"

        else:
            self.log_file_str += "Could not load data" + "\n"
            raise NameError('HiThere')

        #assign datatypes to each column in self.df
        columns_list = self.df.columns.to_list()

        for one_column in columns_list:
            if one_column in int_col_list:
                self.df[one_column] = self.df[one_column].astype('int')
            elif one_column in obj_col_list:
                pass
            else:
                self.df[one_column] = self.df[one_column].astype('float')
        
        print("Loaded data datatypes" + str(self.df.dtypes))
        print("Loaded Data:", self.df.shape)
        self.log_file_str += "Loaded data types successfully" + "\n"

        print("self.df.columns.tolist() = " + str(self.df.columns.tolist()))
    

    def map_categorical(self):
        """Mappings to consolidate categorical variables:"""
        self.df['TITLE_CATEGORY_YR_END'] = self.df['TITLE_CATEGORY_YR_END'].map({'ACCOUNT_EXECUTIVE': 'ACCOUNT_EXECUTIVE', 'ACCOUNT_MANAGER': 'ACCOUNT_EXECUTIVE',
                                    'FIELD_SALES_REPRESENTATIVE': 'REPRESENTATIVE', 'INSIDE_SALES_REPRESENTATIVE': 'REPRESENTATIVE', 'LEAD_GENERATION_REPRESENTATIVE': 'REPRESENTATIVE',
                                    'BUSINESS_DEVELOPMENT_ALLIANCE_PARTNERSHIP_REP': 'REPRESENTATIVE', 'SERVICES_REPRESENTATIVE': 'REPRESENTATIVE',
                                    'CUSTOMER_SUCCESS_REPRESENTATIVE': 'REPRESENTATIVE', 'CHANNEL_SALES_REPRESENTATIVE': 'REPRESENTATIVE',
                                    'SALES_SYSTEMS_APPLICATIONS_ENGINEER': 'CONSULTANT', 'SALES_SUPPORT_OPERATIONS': 'CONSULTANT', 'PRE_SALES_CONSULTANT': 'CONSULTANT',
                                    'PRODUCT_INDUSTRY_SPECIALIST': 'CONSULTANT', 'SALES_MANAGER': 'MANAGER', 'SERVICES_MANAGER': 'MANAGER', 'STRATEGIC_KEY_GLOBAL_ACCOUNT_MANAGER': 'MANAGER',
                                    'MANAGER_SALES_SYSTEMS_APPLICATION_ENGINEERING': 'MANAGER', 'MANAGER_CHANNEL_SALES': 'MANAGER', 'MANAGER_BUSINESS_DEVELOPMENT_ALLIANCE_PARTNERSHIPS': 'MANAGER',
                                    'MANAGER_INSIDE_SALES': 'MANAGER', 'MANAGER_LEAD_GENERATION': 'MANAGER', 'SALES_DIRECTOR': 'DIRECTOR', 'SALES_EXECUTIVE': 'DIRECTOR'})

        self.df['INDUSTRY'] = self.df['INDUSTRY'].map({'Software': 'SaaS & Cloud', 'Media & Internet': 'SaaS & Cloud', 'SaaS & Cloud': 'SaaS & Cloud',
                                    'Financial Services': 'Business Services', 'Business Services': 'Business Services',
                                    'Travel & Hospitality': 'Communications', 'Communications': 'Communications',
                                    'Medical Devices': 'Life Sciences & Pharma','Healthcare': 'Life Sciences & Pharma', 'Life Sciences & Pharma': 'Life Sciences & Pharma',
                                    'Consumer Goods': 'Retail', 'Retail': 'Retail',
                                    'Manufacturing': 'Manufacturing', 'High-Tech Manufacturing': 'Manufacturing',
                                    'Energy': 'Government/Public Sector', 'Other': 'Government/Public Sector', 'Government/Public Sector': 'Government/Public Sector'})
        print("Mapped Categorical:", self.df.shape)
        self.log_file_str += "Mapped categorical data successfully" + "\n"
    
    
    def drop_ids(self):
        """Remove ids for dependent variables where all 0's or 1's:"""

        df = self.df.groupby(['BUSINESS_ID', 'TERM_NEXT_YEAR'])['TERM_NEXT_YEAR'].count()
        business_ids = df.index.get_level_values('BUSINESS_ID')

        seen = {}
        effective_ids = []

        for x in business_ids:
            if x not in seen:
                seen[x] = 1
            else:
                if seen[x] == 1:
                    effective_ids.append(x)
                seen[x] += 1

        ineffective_ids = []
        for x in business_ids:
            if seen[x] == 1:
                ineffective_ids.append(x)

        self.df = self.df.loc[self.df['BUSINESS_ID'].isin(effective_ids)]

        if self.calendar_type == 'fiscal':

            print("For fiscal preds: cols = " + str(self.df.columns.tolist()))
            print('------')

            df = self.df.groupby(['BUSINESS_ID', 'FISCAL_YEAR', 'TERM_NEXT_YEAR'])[['TERM_NEXT_YEAR']].count()
        else:
            df = self.df.groupby(['BUSINESS_ID', 'CAL_YEAR', 'TERM_NEXT_YEAR'])[['TERM_NEXT_YEAR']].count()

        business_ids = np.unique(df.index.get_level_values('BUSINESS_ID'))
        seen = {}
        effective_ids = []
        for x in df.index:
            id_ = x[0]
            cal_year = x[1]
            term = x[2]
            pair = (id_, cal_year)
            if pair not in seen:
                if term == 0:
                    seen[pair] = 2
                    effective_ids.append(pair)
                else:
                    seen[pair] = 1
        
        ineffective_ids = []
        for x in df.index:
            id_ = x[0]
            cal_year = x[1]
            pair = (id_, cal_year)
            if seen[pair] == 1:
                ineffective_ids.append(pair)
        ids = [i[0] for i in ineffective_ids]

        self.df = self.df.loc[~self.df['BUSINESS_ID'].isin(ids)]
        print("Dropped IDs:", self.df.shape)
        self.log_file_str += "Dropped IDs successfully" + "\n"
    
    
    def drop_columns(self):

        global OPTIONAL_REPORTING_COLUMNS

        print("in drop_columns....")
        print("self.df.columns.tolist() = " + str(self.df.columns.tolist()))
        print('--------')

        #first drop mandatory and optional reporting columns - we will re-merge mandatory columns at the end
        self.df_mandatory_reporting_columns = self.df[self.mandatory_reporting_columns_list]
        self.df = self.df.drop(self.mandatory_reporting_columns_list, axis=1)
        
        #not all of the optional reporting columns might be in the file
        optional_columns_in_file_list = []

        for this_opt_column in OPTIONAL_REPORTING_COLUMNS_LIST:
            if this_opt_column in self.df.columns.tolist():
                optional_columns_in_file_list.append(this_opt_column)      
        
        self.df_optional_reporting_columns = self.df[optional_columns_in_file_list]
        self.df = self.df.drop(optional_columns_in_file_list, axis=1)

        # Dropping columns that are not 3/5 full or not used:
        na_col_thresh = len(self.df) * .5
        prev_columns = self.df.columns
        self.df.dropna(axis = 'columns', thresh = na_col_thresh, inplace = True)
        post_columns = self.df.columns
        self.dropped_columns = prev_columns[~(prev_columns.isin(post_columns))]
        print(self.dropped_columns)

        #merge back mandatory reporting_columns
        self.df = pd.concat([self.df_mandatory_reporting_columns, self.df], axis=1)

        print("Dropped Columns:", self.df.shape)
        self.log_file_str += "Just dropped sparse columns successfully" + "\n"

    def one_hot_encode_categorical(self):

        # One Hot Encode on Categorical Data:
        self.num_one_hot = (len(np.unique(self.df['TITLE_CATEGORY_YR_END'].dropna())) + len(np.unique(self.df['INDUSTRY'].dropna())) +
                    len(np.unique(self.df['OWNERSHIP'].dropna())))
        for feature in ['TITLE_CATEGORY_YR_END', 'INDUSTRY', 'OWNERSHIP']:
            self.df = pd.concat([self.df, pd.get_dummies(self.df[feature], prefix = feature)], axis=1)
            self.df.drop([feature], axis = 1, inplace = True)
        print("One Hot Encode:", self.df.shape)
        self.log_file_str += "One-hot encoding completed successfully" + "\n"
    

    def set_dependent_last(self):
        # Set the Dependent variable as last in the set: 
        self.df  = self.df[[c for c in self.df.columns if c not in ['TERM_NEXT_YEAR']] +
                             ['TERM_NEXT_YEAR']]
        print("Set Dependent Last.")
        self.log_file_str += "Set dependent variable last successfully" + "\n"
    
    def seperate_categorical_numeric(self):
        self.tot_col = self.df.shape[1]
        self.one_hot_df = self.df.iloc[:, (self.tot_col-self.num_one_hot-1):(self.tot_col-1)]
        self.var_df = self.df.iloc[:, 0:(self.tot_col-self.num_one_hot-1)]
        self.one_hot_df.reset_index(drop = True, inplace = True)
        print("Seperate Numeric DF, Categorical DF.")
        self.log_file_str += "Separated numeric and categorical DFs successfully" + "\n"

        print("var_df shape:")
        print(self.var_df.shape)
    
    @staticmethod 
    def outlier_quartile_replacement(x):
         
        upper_q = np.nanpercentile(x, 75)
        lower_q = np.nanpercentile(x, 25)
        std = np.nanstd(x)
        mean = np.nanmean(x)
        x = np.where(x >= mean + 3*std, upper_q, x) 
        x = np.where(x <= mean - 3*std, lower_q, x)
        return x
    
    def outlier_removal(self):

        X_old = self.var_df.values

        X = X_old[:, 0:X_old.shape[1] - 1]
        Y = X_old[:,-1].reshape(-1,1)

        X = np.apply_along_axis(self.outlier_quartile_replacement, axis = 0, arr = X)
        X_new = np.concatenate((X, Y), axis = 1)

        self.var_df = pd.DataFrame(X_new, columns = self.var_df.columns)

        print("Outliers Removed:")
        self.log_file_str += "Outliers removed successfully" + "\n"
    
    def reset_index(self):
       
        # Resetting the index for each respective dataframe:
        self.df.reset_index(drop = True, inplace = True)

        idx = self.var_df.index
        self.df = self.df.iloc[idx, :]
        self.one_hot_df = self.one_hot_df.iloc[idx, :]
        
        self.df.reset_index(drop = True, inplace = True)
        self.one_hot_df.reset_index(drop = True, inplace = True)
        self.var_df.reset_index(drop = True, inplace = True)

        print("Indeces Reset.")
        print("Number of rows at end of reset_index: " + str(len(self.df)))
        self.log_file_str += "Indices reset successfully" + "\n"
        

    def split_data(self):

        [self.train, self.test] = train_test_split(self.var_df, test_size = 1/3, random_state = 42)
        self.train_idx = self.train.index
        self.test_idx = self.test.index
        print("Data split.")
        self.log_file_str += "Data split successfully" + "\n"
        

    def add_difference_variables(self):
        # Difference Variables:
        self.train['DIFF_MONTH_ORDER'] = self.train['COUNT_MAX_MONTH_ORDER'] - self.train['COUNT_MIN_MONTH_ORDER']
        self.test['DIFF_MONTH_ORDER'] = self.test['COUNT_MAX_MONTH_ORDER'] - self.test['COUNT_MIN_MONTH_ORDER']

        self.train['DIFF_QUOTA_AMT_USD'] = self.train['MAX_QUOTA_AMT_USD'] - self.train['MIN_QUOTA_AMT_USD']
        self.test['DIFF_QUOTA_AMT_USD'] = self.test['MAX_QUOTA_AMT_USD'] - self.test['MIN_QUOTA_AMT_USD']
        print("Difference Variables Created.")

        self.log_file_str += "Difference variables created successfully" + "\n"

    def impute_data(self):

        print("In impute data")

        npa_train_mandatory_reporting_columns = self.train[self.mandatory_reporting_columns_list].values
        self.train = self.train.drop(self.mandatory_reporting_columns_list, axis=1)

        train_columns_list = self.major_train_columns_list + self.train.columns.tolist()

        npa_test_mandatory_reporting_columns = self.test[self.mandatory_reporting_columns_list].values
        self.test = self.test.drop(self.mandatory_reporting_columns_list, axis=1)

        test_columns_list = self.major_test_columns_list + self.test.columns.tolist()

        # Iterative approach to filling in null values:
        x_train = self.train.values
        x_test = self.test.values

        self.imp = IterativeImputer(max_iter = 10, random_state = 42, initial_strategy = 'mean', estimator = BayesianRidge())
        x_train = self.imp.fit_transform(x_train)
        x_test = self.imp.transform(x_test)
        #self.imp = x_train.mean(axis = 0)

        #prepend reporting columns as NumPY arrays using hstack
        x_train = np.hstack((npa_train_mandatory_reporting_columns, x_train))
        x_test = np.hstack((npa_test_mandatory_reporting_columns, x_test))

        self.train = pd.DataFrame(x_train, columns = train_columns_list)
        self.test = pd.DataFrame(x_test, columns = test_columns_list)

        print("Data Imputed.")
        self.log_file_str += "Data imputed successfully" + "\n"

    def recombine_data(self):

        # Merge Complete Numeric df with complete Categorical df:
        self.tot_col = self.df.shape[1]
        one_hot_train = self.one_hot_df.iloc[self.train_idx,:]
        one_hot_test = self.one_hot_df.iloc[self.test_idx,:]
        one_hot_train.reset_index(drop = True, inplace = True)
        one_hot_test.reset_index(drop = True, inplace = True)
        
        self.train = pd.merge(self.train, one_hot_train, left_index=True, right_index=True)
        self.test = pd.merge(self.test, one_hot_test, left_index=True, right_index=True)

        # Block code below in cell for Dependent Inclusive approach to null values:
        dep_train = self.df.iloc[self.train_idx, self.tot_col-1]
        dep_train.reset_index(drop = True, inplace = True)
        self.train['TERM_NEXT_YEAR'] = dep_train

        dep_test = self.df.iloc[self.test_idx, self.tot_col-1]
        dep_test.reset_index(drop = True, inplace = True)
        self.test['TERM_NEXT_YEAR'] = dep_test
        print(self.train.info())
        print(self.test.info())
        print('Remerged Dataframes.')
        self.log_file_str += "Remerged dataframes successfully" + "\n"

    @staticmethod
    def normalize_x(self, x_train, x_test, ignore_cols):
        train_copy = np.copy(x_train)
        test_copy = np.copy(x_test)
    
        train_copy = np.delete(train_copy, ignore_cols, axis = 1)
        test_copy = np.delete(test_copy, ignore_cols, axis = 1)
    
        self.norm_scaler = StandardScaler()
        train_copy = self.norm_scaler.fit_transform(train_copy)
        test_copy = self.norm_scaler.transform(test_copy)
    
        for i in ignore_cols:
            train_copy = np.insert(train_copy, i, x_train[:, i], axis = 1)
            test_copy = np.insert(test_copy, i, x_test[:, i], axis = 1)
        x_train, x_test = train_copy, test_copy
        return x_train, x_test

    def normalize_data(self):

        # Split the Data/Normalize based off of percentage:
        self.tot_col = self.train.shape[1]
        X_ = self.train.values
        self.x_train = X_[:, 0:X_.shape[1] - 1]
        self.y_train = X_[:,-1]

        X_ = self.test.values
        self.x_test = X_[:, 0:X_.shape[1] - 1]
        self.y_test = X_[:,-1]

        ignore = np.arange((self.tot_col-self.num_one_hot-1), (self.tot_col-1))
        
        #add the 4 reporting columns to the list of ignored columns 
        ignore = np.concatenate((np.array([0,1,2,3]), ignore))
        
        self.x_train, self.x_test = self.normalize_x(self, self.x_train, self.x_test, ignore_cols = ignore)

        # Recreate Train and Test to query out more columns (if needed):
        columns = self.train.iloc[:,0:len(self.train.columns)-1].columns
        self.train = pd.DataFrame(self.x_train, columns = columns)
        self.test = pd.DataFrame(self.x_test, columns = columns)

        print("Normalized Data.")
        self.log_file_str += "Normalized data successfully" + "\n"


    def select_features(self):

        global TEST_MODE_ON

        selected_feat = ['PR_TARGET_USD', 'SALARY_USD', 'COUNT_MONTH_EMPLOYED_TIL_DEC',
                    'COUNT_MONTHS_GOT_PAYMENT', 'LAST_PAYMENT_UNTIL_YR_END', 'YEAR_PAYMENT',
                    'COUNT_UNIQ_QUOTA', 'COUNT_AVG_MONTH_PAID_QUOTA',
                    'LAST_QUOTA_PAID_UNTIL_YR_END', 'SUM_CREDIT_AMT_USD',
                    'MIN_CREDIT_AMT_USD', 'DIFF_QUOTA_AMT_USD', 'COUNT_UNIQ_TITLE_NAME', 'COUNT_UNIQ_MGR_ID']
        
        #do we need this as a selected feature?
        if self.calendar_type == 'gregorian':
            selected_feat.append('COMP_AVG_MONTH_PAYEECOUNT')

        categ_col = self.train.iloc[:, (len(self.train.columns) - self.num_one_hot):]

        self.features = np.concatenate((self.mandatory_reporting_columns_list, selected_feat, categ_col.columns))

        # Query out Selected Feat (if wanted):
        self.train = self.train[self.features]
        self.test = self.test[self.features]

        self.x_train = self.train.values
        self.x_test = self.test.values
        
        self.train['TERM_NEXT_YEAR'] = self.y_train
        self.test['TERM_NEXT_YEAR'] = self.y_test
        self.df = pd.concat([self.train, self.test], axis = 0)

        #write data to DBFS so that LivePrediction class can read it, and we don't have to do imputation again
        cleaned_file_name = ''
        if self.calendar_type == 'gregorian':
            cleaned_file_name = 'cleaned_gregorian_data.tsv'
        elif self.calendar_type == 'fiscal':
            cleaned_file_name = 'cleaned_fiscal_data.tsv'

        self.df.to_csv('/dbfs/FileStore/attrition_test_data/' + cleaned_file_name, index=False, sep='\t')

        print("Selected Features.")
        print(self.df)
        self.log_file_str += "Selected features successfully" + "\n"
    
    def save_data_object(self):
        self.df = None
        self.num_one_hot = None
        self.one_hot_features = None
        self.tot_col = None
        self.var_df = None
        self.train = None
        self.test = None
        self.train_idx = None
        self.test_idx = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.nightly_file = None
        if self.calendar_type == 'fiscal':
            path = '/dbfs/FileStore/pickleFiles/data_object_fiscal.obj'
            file_out = open(path, 'wb')
            dump(self, file_out)
            self.log_file_str += "Saved fiscal data object successfully" + "\n"
        elif self.calendar_type == 'gregorian':
            path = '/dbfs/FileStore/pickleFiles/data_object_gregorian.obj'
            file_out = open(path, 'wb')
            dump(self, file_out)
            self.log_file_str += "Saved gregorian data object successfully" + "\n"
        else:
            raise ValueError("Wrong name.")
    
    def get_log_file(self):
        return self.log_file_str

# COMMAND ----------

class RandomForest(object):
    def __init__(self, CleanData, model_type):
        self.x_train = CleanData.x_train
        self.x_test = CleanData.x_test
        self.y_train = CleanData.y_train
        self.y_test = CleanData.y_test
        self.df = CleanData.df
        self.calendar_type = CleanData.calendar_type
        self.model_type = model_type
        self.rf = None
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        timestamp = datetime.fromtimestamp(timestamp)
        self.nightly_file = open("nightly.txt", "a")
        self.nightly_file.write("\n" + "Timestamp: " + str(timestamp))
        self.nightly_file.write(", Calendar Type - " + self.calendar_type + ", Model Type - " + self.model_type)
        
    def predict(self):

        npa_reporting_columns = None
        npa_reporting_col_data_test = None

        "Creation of model based on fitting on training data, predicting on testing data."
        if self.model_type == 'DecisionTree':
            self.dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 100, min_samples_split = 20, ccp_alpha = 0.00001)
            self.dt.fit(self.x_train, self.y_train)
            y_test_pred = self.dt.predict(self.x_test)
            train_score = self.dt.score(self.x_train, self.y_train)
            test_score = self.dt.score(self.x_test, self.y_test)
            auc_score = self.dt.score(self.x_test, self.y_test)
            print("Test score: ", test_score, "Train score: ", train_score)
        elif self.model_type == 'RandomForest':
            # Create Random Forest Model:
            self.rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = 50, max_features = 20, min_samples_split = 10, ccp_alpha=0.00001)
            
            #we need to separate out the reporting columns before doing any fitting
            npa_reporting_col_data_train = self.x_train[:, :4]
            self.x_train = self.x_train[:, 4:]
            npa_reporting_col_data_test = self.x_test[:, :4]
            self.x_test = self.x_test[:, 4:]

            #reshape y_train and y_test
            self.y_train = self.y_train.reshape(-1,1)
            self.y_test = self.y_test.reshape(-1,1)
            
            print("calendar type is: " + str(self.calendar_type))
        
            self.rf.fit(self.x_train, self.y_train)
            y_test_pred = self.rf.predict(self.x_test)
            train_score = self.rf.score(self.x_train, self.y_train)
            test_score = self.rf.score(self.x_test, self.y_test)
            print("Test score: ", test_score, "Train score: ", train_score)

            print("Predictions executed successfully!")

            #add reporting cols back if necessary
            self.x_train = np.hstack((npa_reporting_col_data_train, self.x_train))
            self.x_test = np.hstack((npa_reporting_col_data_test, self.x_test))
            print("Predict activities finished!")
              
        self.nightly_file.write("Training Score: " + str(train_score) + ", Test Score: " + str(test_score)) 
        self.auc()

    def auc(self):

        npa_reporting_col_data_test = None

        if self.model_type == 'DecisionTree':
            y_test_probs = self.dt.predict_proba(self.x_test)    
            y_test_binary = label_binarize(self.y_test, classes = [0, 1])
            y_test_probs = np.array(pd.DataFrame(y_test_probs)[1]).reshape(-1,1)
            auc = roc_auc_score(y_test_binary, y_test_probs)
        elif self.model_type == 'RandomForest':

            #strip out reporting columns, and then re-merge them at bottom
            npa_reporting_col_data_test = self.x_test[:, :4]
            self.x_test = self.x_test[:, 4:]

            y_test_probs = self.rf.predict_proba(self.x_test)    
            y_test_binary = label_binarize(self.y_test, classes = [0, 1])
            y_test_probs = np.array(pd.DataFrame(y_test_probs)[1]).reshape(-1,1)
            auc = roc_auc_score(y_test_binary, y_test_probs)

            #re-merge reporting columns
            self.x_test = np.hstack((npa_reporting_col_data_test, self.x_test))

        print("AUC: " + str(auc))
        self.nightly_file.write(", AUC: " + str(auc))
        self.nightly_file.close()

    def predict_final(self):
        """Need to have model train on entire dataset."""
        
        #global TEST_MODE_ON
        #global MANDATORY_REPORTING_COLUMNS_LIST

        #df_mandatory_reporting_columns = None

        #strip out reporting columns
        #if TEST_MODE_ON == True:
        #df_mandatory_reporting_columns = self.df[MANDATORY_REPORTING_COLUMNS_LIST]
        #self.df = self.df.drop(MANDATORY_REPORTING_COLUMNS_LIST, axis=1)
 
        # All data needs to be trained on:
        X = self.df.drop(['TERM_NEXT_YEAR'], axis = 1).values
        Y = self.df['TERM_NEXT_YEAR'].values

        #remove reporting columns
        npa_reporting_col_data_test = X[:, :4]
        X = X[:, 4:]

        #round all values in X to 3 digits after the decimal point
        X = np.round(X, 3)

        #self.df.to_csv("/dbfs/FileStore/attrition_test_data/self_df_predict_final.csv", index=False)

        if self.model_type == 'DecisionTree':
            self.dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 100, min_samples_split = 20, ccp_alpha = 0.00001)
            self.dt.fit(X, Y)
        elif self.model_type == 'RandomForest':
            self.rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = 50, max_features = 20, min_samples_split = 10, ccp_alpha=0.00001)
            self.rf.fit(X, Y)
        
        #re-merge reporting columns
        #self.df = pd.concat([df_mandatory_reporting_columns, self.df], axis=1)

    def save_model(self):
        """Saves the model as a pickle file to be utilized."""
        if self.calendar_type == 'fiscal':
            path = '/dbfs/FileStore/pickleFiles/rf_model_fiscal.pkl'
            file_out = open(path, 'wb')
            dump(self.rf, file_out)
        elif self.calendar_type == 'gregorian':
            path = '/dbfs/FileStore/pickleFiles/rf_model_gregorian.pkl'
            file_out = open(path, 'wb')
            dump(self.rf, file_out)
        else:
            raise ValueError("Wrong path")

# COMMAND ----------

class LivePrediction(object):
    """ Creates a prediction object that can then be used to clean the data and output prediction. """

    def __init__(self, type_model, header_path, data_path, pred_output_path, visual_output_path, model_path, fortnightly_prediction_output_path, fortnightly_log_file_path, log_file_str):
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

        #we will log events using a string
        self.log_file_str = log_file_str
        self.log_file_str += "Start attrition model run: " + datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '\n'
        self.log_file_str += "====================" + "\n"

        try:
        
            self.df = pd.read_csv(self.data_path, sep='\t')

            print(self.df.columns.tolist())

            if self.df.shape[1] == MAX_COL:
                self.df = self.df.iloc[:, 0:-2]

            self.log_file_str += 'Input file loaded successfully' + '\n'

        except FileNotFoundError as f:
            msg = "Data Path or Header Path Incorrect."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(f)
            raise FileNotFoundError(msg)
        
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
                    open(self.model_path + '/rf_model_fiscal.pkl', 'rb')
                )

                self.log_file_str += 'rf_model_fiscal.pkl successfully loaded' + '\n'

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
                    open(self.model_path + '/rf_model_gregorian.pkl', 'rb')
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
    
        
    def save_reporting_columns(self):
        
        self.business_ids = self.df['BUSINESS_ID']
        self.master_participant_id = self.df['MASTER_PARTICIPANT_ID']
        self.term_as_of_date = self.df['TERM_AS_OF_DATE']

        if self.type_model == "gregorian":
            self.cal_year = self.df['CAL_YEAR']
        else:
            self.cal_year = self.df['FISCAL_YEAR']
        
        self.log_file_str += 'Saved all reporting columns' + '\n'


    def clean_input(self):
        """ Cleans the input in the same manner as the training of the model. """

        global TEST_MODE_ON

        #Drop unused features, save to object BUSINESS_ID, MASTER_PARTICIPANT_ID:
        try:

            self.business_ids = self.df['BUSINESS_ID']
            self.master_participant_id = self.df['MASTER_PARTICIPANT_ID']
            self.term_as_of_date = self.df['TERM_AS_OF_DATE']
            
            if self.type_model == 'fiscal':
                self.fiscal_year = self.df['CAL_YEAR']

                self.df.drop(
                    ['BUSINESS_ID', 'TERM_AS_OF_DATE','HOME_CITY', 'HOME_STATE_PROVINCE', 'HOME_COUNTRY_CODE', 
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

            elif self.type_model == 'gregorian':

                self.cal_year = self.df['CAL_YEAR']

                print("About to drop BUSINESS_ID and other cols in gregorian df...")

                self.df.drop(
                    ['BUSINESS_ID', 'TERM_AS_OF_DATE', 'HOME_CITY', 'HOME_STATE_PROVINCE', 'HOME_COUNTRY_CODE',
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
        

        try:
            pass
            # imp = IterativeImputer(max_iter = 10, random_state = 42, initial_strategy = 'mean', estimator = BayesianRidge())
            # var_array = imp.transform(var_df.values)

            # norm_scaler = StandardScaler()
            # var_array = norm_scaler.fit_transform(var_array)

            # var_array = self.data_object.imp.transform(
            #    var_df.values
            # )
            # var_array = self.data_object.norm_scaler.transform(
            #    var_array
            # )
        except ValueError as v:
            msg = "Array passed into imputer or normalizer are incorrect lengths."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(v)
            raise ValueError(msg)

        #var_df = pd.DataFrame(
        #    var_array, columns=var_df.columns
        #)
        
        
        self.df = pd.concat(
            [var_df, one_hot_df], axis=1
        )

        #Ensure all Categorical Features Present:
        try:

            for col in self.data_object.one_hot_df.columns:
                if col not in one_hot_features:
                    self.df[col] = 0

        except AttributeError as a:
            msg = "Incompatible versions of Pandas between Pickle File and VM."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(a)
            raise AttributeError(msg)

        #Feature Select/Align Order:
        if TEST_MODE_ON == True:
            print('test file features are:')
            print(self.df.columns.tolist())
            self.df = self.df[self.data_object.features]
        
        print("Cleaned Data Successfully.")

    def predict(self):

        global TEST_MODE_ON
        global MANDATORY_REPORTING_COLUMNS_LIST

        self.log_file_str += '\n' + 'In predict()' + '\n'
        self.log_file_str += '====================' + '\n'

        reporting_column_list = MANDATORY_REPORTING_COLUMNS_LIST.copy()

        """ Predict the Attrition likelihood for input data and outputs the results as a tsv. """
        try:
            
            if self.type_model == "fiscal":
                cal_year_idx = reporting_column_list.index("CAL_YEAR")
                reporting_column_list[cal_year_idx] = "FISCAL_YEAR"

            df_mandatory_reporting_columns = self.df[reporting_column_list]
            self.df = self.df.drop(reporting_column_list, axis=1)

            rf_model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = 50, max_features = 20, min_samples_split = 10, ccp_alpha=0.00001)

            self.log_file_str += 'Successfully created RandomForestClassifier' + '\n'

            #model training/testing
            [df_train, df_test] = train_test_split(self.df, test_size = 1/3, random_state = 42)

            df_train_target = df_train[['TERM_NEXT_YEAR']]
            df_train = df_train.drop(['TERM_NEXT_YEAR'], axis=1)

            df_test_target = df_test[['TERM_NEXT_YEAR']]
            df_test = df_test.drop(['TERM_NEXT_YEAR'], axis=1)

            x_train = df_train.values
            y_train = df_train_target.values

            x_test = df_test.values
            y_test = df_test_target.values

            #train model
            rf_model.fit(x_train, y_train)
            y_test_pred = rf_model.predict(x_test)

            self.log_file_str += 'Successfully trained RandomForestClassifier' + '\n'

            train_score = rf_model.score(x_train, y_train)
            test_score = rf_model.score(x_test, y_test)

            print("Test score: ", test_score, "Train score: ", train_score)
            self.log_file_str += 'Train score: ' + str(train_score) + '\n'
            self.log_file_str += 'Test score: ' + str(test_score) + '\n'

            #generate preds and probs for self.df
            #first, extract target
            self.df = self.df.drop(['TERM_NEXT_YEAR'], axis=1)

            probs = rf_model.predict_proba(self.df.values)

            self.log_file_str += 'Generated probs successfully' + '\n'
          
            #print('Executed predict_proba successfully')


        except ValueError as v:
            exception_str = repr(v)
            msg = "Array passed into model is of incorrect length."
            print(exception_str)
            self.log_file_str += msg + ' ' + exception_str
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(v)
            raise ValueError(msg)
        except:
            msg = "Unspecified Error when making prediction."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            raise ValueError(msg)

        if self.type_model == 'fiscal':

            d = {
                'BUSINESS_ID': self.business_ids, 'MASTER_PARTICIPANT_ID': self.master_participant_id,
                'FISCAL_YEAR': self.cal_year, 'PRED_TERM_PROB': probs[:, 1]
                }
            
            try:
                
                self.results = pd.DataFrame(d)

                #convert BUSINESS_ID, MASTER_PARTICIPANT_ID and FISCAL_YEAR to type int
                self.results.columns = ['BUSINESS_ID', 'MASTER_PARTICIPANT_ID', 'FISCAL_YEAR', 'PRED_TERM_PROB']
                columns_to_convert = ['BUSINESS_ID', 'MASTER_PARTICIPANT_ID', 'FISCAL_YEAR']
                
                self.results[columns_to_convert] = self.results[columns_to_convert].astype(int)

                if TEST_MODE_ON == True:
                    self.results.to_csv(self.pred_output_path + "results_test_fiscal.csv", index=False)
                    self.log_file_str += 'Generated results_test_fiscal.csv successfully: ' + '\n'
                else:

                    df_spark = spark.createDataFrame(self.results)

                    df_spark.write \
                        .format("snowflake") \
                        .mode("overwrite") \
                        .options(**options) \
                        .option("dbtable", "TURNOVER_PRED_FISCAL") \
                        .save()
                    
                    self.log_file_str += 'Wrote self.results to Snowflake successfully: ' + '\n'
                
                #get today's date and time as a string
                current_datetime_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

                #write data to Databricks as well, whether we are using a test file or not
                self.results.to_csv(
                    self.fortnightly_prediction_output_path + '/part_fiscal_pred_' + current_datetime_str + '.tsv',
                    index=False, sep='\t'
                )

                self.log_file_str += 'Wrote fortnightly_prediction file successfully: ' + '\n'
                
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
                self.results.columns = ['BUSINESS_ID', 'MASTER_PARTICIPANT_ID', 'CAL_YEAR', 'PRED_TERM_PROB']

                #convert BUSINESS_ID, MASTER_PARTICIPANT_ID and CAL_YEAR to type int
                columns_to_convert = ['BUSINESS_ID', 'MASTER_PARTICIPANT_ID', 'CAL_YEAR']

                self.results[columns_to_convert] = self.results[columns_to_convert].astype(int)
                
                if TEST_MODE_ON == True:
                    self.results.to_csv(self.pred_output_path + "results_test_gregorian.csv", index=False)
                    self.log_file_str += 'Generated results_test_gregorian.csv successfully: ' + '\n'
                else:


                    df_spark = spark.createDataFrame(self.results)

                    df_spark.write \
                        .format("snowflake") \
                        .mode("overwrite") \
                        .options(**options) \
                        .option("dbtable", "TURNOVER_PRED_GREGORIAN") \
                        .save()
                    
                    self.log_file_str += 'Wrote self.results to Snowflake successfully: ' + '\n'
                
                #get today's date and time as a string
                current_datetime_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

                #write data to Databricks as well
                self.results.to_csv(
                   self.fortnightly_prediction_output_path + '/part_gregorian_pred_' + current_datetime_str + '.tsv',
                   index=False, sep='\t'
                )

                self.log_file_str += 'Wrote fortnightly_prediction file successfully: ' + '\n'
               
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
        global TEST_MODE_ON

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

        except IndexError as i:
            msg = "No shap values were created from the TreeExplainer."
            self.log_file_str += msg
            self.write_log_file()
            #logger.critical(msg)
            #logger.critical(i)
            raise IndexError(msg)

        shap_results = pd.DataFrame(
            shap_arr, columns=self.df.columns
        )
        self.results = pd.concat(
            [self.results, shap_results], axis=1
        )
        #Combine Shap Values of One-Hot-Encoded Features:
        for prefix in ['TITLE_CATEGORY_YR_END', 'INDUSTRY', 'OWNERSHIP']:
            filter_col = [col for col in self.results.columns if col.startswith(prefix)]
            self.results[prefix] = self.results[filter_col].sum(axis=1)
            self.results.drop(
                filter_col,
                axis=1, inplace=True
            )
        self.results["EXPECTED_VALUE"] = shap_exp.expected_value[1]

        self.log_file_str += 'combined shap values with one-hot-encoded features successfully' + '\n'

        #if self.type_model == 'gregorian':
        #    del self.results["COMP_AVG_MONTH_PAYEECOUNT"]

        #get today's date and time as a string
        current_datetime_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

        self.results.to_csv(
            self.visual_output_path + '/part_visual_' + self.type_model + "_" + current_datetime_str +  '.tsv',
            index=False, sep='\t'
        )

        self.log_file_str += 'wrote shap output file successfully' + '\n'

        if TEST_MODE_ON == False:

            snowflake_table_name = ""

            if self.type_model == 'fiscal':
                snowflake_table_name = "VISUALISATION_FISCAL"
            else:
                snowflake_table_name = "VISUALISATION_GREGORIAN"
            
            df_spark = spark.createDataFrame(self.results)

            df_spark.write \
                .format("snowflake") \
                .mode("overwrite") \
                .options(**options) \
                .option("dbtable",snowflake_table_name) \
                .save()

        print("Explainer ended.")
        self.log_file_str += 'Explainer ended' + '\n'

# COMMAND ----------

#read the TRIGGER_ATTRITION table
df_trigger_attrition = spark.read \
                    .format("snowflake") \
                    .options(**options) \
                    .option("dbtable","INSIGHTS_PARAMETER") \
                    .load().toPandas()

#if the flag is False,then we don't execute attrition
trigger_attrition_flag = df_trigger_attrition.loc[0, 'VALUE']

print(trigger_attrition_flag)

if trigger_attrition_flag == 'FALSE' or TEST_MODE_ON == True:

    #create cleaning object for gregorian predictions - but only for real files
    gregorian_data = CleanData('gregorian', test_mode=TEST_MODE_ON, log_file_str=gregorian_log_file_str)
    rfModelGregorian = RandomForest(gregorian_data, 'RandomForest')
    rfModelGregorian.predict()
    rfModelGregorian.predict_final()
    rfModelGregorian.save_model()
    gregorian_data.save_data_object()

    print("Finished all gregorian stuff, on to fiscal stuff....")

    #create cleaning object for fiscal predictions
    fiscal_data = CleanData('fiscal', test_mode=TEST_MODE_ON, log_file_str=fiscal_log_file_str)
    rfModelFiscal = RandomForest(fiscal_data, 'RandomForest')
    rfModelFiscal.predict()
    rfModelFiscal.predict_final()
    rfModelFiscal.save_model()
    fiscal_data.save_data_object()

    print("all done with cleaning!")
    print("let's start making predictions....")
    print('----------')

    #get log files from CleanData objects
    fiscal_log_file_str = fiscal_data.get_log_file()
    gregorian_log_file_str = gregorian_data.get_log_file()

    #add line breaks for prediction portion of logging statements
    fiscal_log_file_str += "\n" + "====================" + "\n"
    gregorian_log_file_str += "\n" + "====================" + "\n"

    #execute both fiscal and gregorian attrition

    fiscal_args = ["fiscal", fiscal_header_path_cl, fiscal_file_path_cl, fiscal_output_path, fiscal_vis_output_path, model_input_cl, fortnightly_prediction_output_path, fortnightly_log_file_path, fiscal_log_file_str]
    gregorian_args = ["gregorian", gregorian_header_path_cl, gregorian_file_path_cl, gregorian_output_path, gregorian_vis_output_path, model_input_cl, fortnightly_prediction_output_path, fortnightly_log_file_path, gregorian_log_file_str]

    for input_lst in [gregorian_args, fiscal_args]:
        batch_pred = LivePrediction(
            input_lst[0], input_lst[1], input_lst[2],
            input_lst[3], input_lst[4], input_lst[5], 
            input_lst[6], input_lst[7], input_lst[8]
        )

        #batch_pred.clean_input()
        batch_pred.save_reporting_columns()
        batch_pred.predict()
        batch_pred.explain_pred()

        #write log file
        batch_pred.write_log_file()
    
    if TEST_MODE_ON == False:
    
        #set trigger_attrition_flag to FALSE
        df_trigger_attrition.loc[0, 'VALUE'] = 'FALSE'

        #convert df_trigger_attrition to a spark dataframe
        df_spark_trigger_attrition = spark.createDataFrame(df_trigger_attrition)

        df_spark_trigger_attrition.write \
            .format("snowflake") \
            .mode("overwrite") \
            .options(**options) \
            .option("dbtable", "INSIGHTS_PARAMETER") \
            .save()
else:

    print("flag is set to FALSE, attrition will not execute")

# COMMAND ----------


