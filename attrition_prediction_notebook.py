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

#read fiscal attrition data
df_attrition = spark.read.csv("dbfs:/FileStore/fiscal_attrition_with_headers.tsv", header=True, sep='\t').toPandas()

MAX_COL = 56

 #save cols into mandatory_reporting_columns and optional_reporting_columns for proper file processing within CleanData
OPTIONAL_REPORTING_COLUMNS_LIST = ['HOME_CITY', 'HOME_STATE_PROVINCE', 'HOME_COUNTRY_CODE','HIRE_AS_OF_DATE', 'TITLE_NAME_YR_END']
MANDATORY_REPORTING_COLUMNS_LIST = ['BUSINESS_ID', 'CAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID']

# COMMAND ----------

# try:
#     dt = datetime.now()
#     logging.basicConfig(filename="/dbfs/FileStore/logs/status" + "_" +  dt.strftime("%Y_%m_%d") + ".log", 
#                         format='%(asctime)s - %(levelname)s - %(message)s', filemode='a', level=logging.ERROR)
#     logger = logging.getLogger()
# except FileNotFoundError as f:
#     msg = "Logger filepath not found."
#     raise FileNotFoundError(msg)

# COMMAND ----------

#constants to control how this notebook is run
TEST_MODE_ON = True
test_fiscal_data_path = ''
test_gregorian_data_path = ''

# COMMAND ----------

#we don't need to supply command line parameters in the Databricks environment
#we also might not need these -- hmmm...

gregorian_header_path_cl = ''
gregorian_file_path_cl = '/dbfs/FileStore/attrition_test_data/cleaned_fiscal_data.tsv'
fiscal_header_path_cl = ''
fiscal_file_path_cl = '/dbfs/FileStore/attrition_test_data/cleaned_gregorian_data.tsv'
gregorian_output_path = '/dbfs/FileStore/prediction_output/'
gregorian_vis_output_path = '/dbfs/FileStore/visual_output/'
fiscal_output_path = '/dbfs/FileStore/prediction_output/'
fiscal_vis_output_path = '/dbfs/FileStore/visual_output/'
model_input_cl = '/dbfs/FileStore/pickleFiles'

# COMMAND ----------

# Snowflake credentials
user = "app_datascience"
password = "Xactly123"

# COMMAND ----------

#Connect to Snowflake

options = {
  "sfUrl": "https://xactly-xactly_engg_datalake.snowflakecomputing.com/",
  "sfUser": user,
  "sfPassword": password,
  "sfDatabase": "XTLY_ENGG",
  "sfSchema": "INSIGHTS",
  "sfWarehouse": "DIS_LOAD_WH"
}

# COMMAND ----------

class CleanData(object):

    def __init__(self, calendar_type, test_mode=False):
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

    def load_data(self):
        """Read in the data:"""

        global TEST_MODE_ON

        int_col_list = ['MASTER_PARTICIPANT_ID','CAL_YEAR','HIRE_AS_OF_DATE','COUNT_UNIQ_TITLE_NAME','COUNT_UNIQ_MGR_ID', 'COUNT_MONTHS_GOT_PAYMENT', 'TERM_NEXT_YEAR']

        obj_col_list = ['TITLE_NAME_YR_END','TITLE_CATEGORY_YR_END','HOME_CITY','HOME_STATE_PROVINCE','HOME_COUNTRY_CODE',
                        'INDUSTRY','OWNERSHIP']

        if self.calendar_type == 'fiscal':

            if TEST_MODE_ON == True:
                self.df = pd.read_csv('/dbfs/FileStore/attrition_test_data/gregorian_attrition.tsv', sep='\t')
            else:
                #read attrition data from Snowflake
                self.df = spark.read \
                        .format("snowflake") \
                        .options(**options) \
                        .option("dbtable","MERGE_FEATURE_FISCAL") \
                        .load().toPandas()
            
            print("Fiscal Data")
            print("------------")
            print("Shape: " + str(self.df.shape))
            print("------------")
            print(self.df.columns.tolist())
            print("------------")
            

            dtypes_series = self.df.dtypes

            dtypes_dict = dtypes_series.apply(lambda x: x.name).to_dict()
            self.df = self.df.astype(dtypes_dict)

            self.df['BUSINESS_ID'] = self.df['BUSINESS_ID'].astype('int')           
            
            print('Loaded Fiscal Data')
        
        elif self.calendar_type == 'gregorian':

            if TEST_MODE_ON == True:
                self.df = pd.read_csv('/dbfs/FileStore/attrition_test_data/fiscal_attrition.tsv', sep='\t')
            else:
                self.df = spark.read \
                        .format("snowflake") \
                        .options(**options) \
                        .option("dbtable","MERGE_FEATURE_GREGORIAN") \
                        .load().toPandas()
            
            print("Gregorian Data")
            print("------------")
            print("Shape: " + str(self.df.shape))
            print("------------")
            print(self.df.columns.tolist())
            print("------------")

            dtypes_series = self.df.dtypes

            dtypes_dict = dtypes_series.apply(lambda x: x.name).to_dict()
            self.df = self.df.astype(dtypes_dict)

            print('Loaded Gregorian Data')
        else:
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
    
    def drop_ids(self):
        """Remove ids for dependent variables where all 0's or 1's:"""
        print("df shape at top:" + str(self.df.shape))

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
            df = self.df.groupby(['BUSINESS_ID', 'CAL_YEAR', 'TERM_NEXT_YEAR'])[['TERM_NEXT_YEAR']].count()
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
    
    def drop_columns(self):

        global MANDATORY_REPORTING_COLUMNS
        global OPTIONAL_REPORTING_COLUMNS

        #first drop mandatory and optional reporting columns - we will re-merge mandatory columns at the end
        self.df.to_csv("/dbfs/FileStore/attrition_test_data/mandatory_col_check_start.csv", index=False)

        self.df_mandatory_reporting_columns = self.df[MANDATORY_REPORTING_COLUMNS_LIST]
        self.df = self.df.drop(MANDATORY_REPORTING_COLUMNS_LIST, axis=1)
        
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

        #self.df.drop(['BUSINESS_ID', 'HIRE_AS_OF_DATE', 'TITLE_NAME_YR_END','CAL_YEAR', 'MASTER_PARTICIPANT_ID'], axis = 1, inplace = True)
        #print("Dropped Columns:", self.df.shape)

    def one_hot_encode_categorical(self):

        # One Hot Encode on Categorical Data:
        self.num_one_hot = (len(np.unique(self.df['TITLE_CATEGORY_YR_END'].dropna())) + len(np.unique(self.df['INDUSTRY'].dropna())) +
                    len(np.unique(self.df['OWNERSHIP'].dropna())))
        for feature in ['TITLE_CATEGORY_YR_END', 'INDUSTRY', 'OWNERSHIP']:
            self.df = pd.concat([self.df, pd.get_dummies(self.df[feature], prefix = feature)], axis=1)
            self.df.drop([feature], axis = 1, inplace = True)
        print("One Hot Encode:", self.df.shape)

    def set_dependent_last(self):
        # Set the Dependent variable as last in the set: 
        self.df  = self.df[[c for c in self.df.columns if c not in ['TERM_NEXT_YEAR']] +
                             ['TERM_NEXT_YEAR']]
        print("Set Dependent Last.")
    
    def seperate_categorical_numeric(self):
        self.tot_col = self.df.shape[1]
        self.one_hot_df = self.df.iloc[:, (self.tot_col-self.num_one_hot-1):(self.tot_col-1)]
        self.var_df = self.df.iloc[:, 0:(self.tot_col-self.num_one_hot-1)]
        self.one_hot_df.reset_index(drop = True, inplace = True)
        print("Seperate Numeric DF, Categorical DF.")

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
        print("Number of rows at end of outlier_removal: " + str(len(self.df)))
    
    def reset_index(self):
        print('In reset index...')
       
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
        

    def split_data(self):
        [self.train, self.test] = train_test_split(self.var_df, test_size = 1/3, random_state = 42)
        self.train_idx = self.train.index
        self.test_idx = self.test.index
        print("Data split.")

        print("Size of self.train: " + str(len(self.train)))
        print("Size of self.test: " + str(len(self.test)))
        print("Size of self.train_idx: " + str(len(self.train_idx)))
        print("Size of self.test_idx: " + str(len(self.test_idx)))
        

    def add_difference_variables(self):
        # Difference Variables:
        self.train['DIFF_MONTH_ORDER'] = self.train['COUNT_MAX_MONTH_ORDER'] - self.train['COUNT_MIN_MONTH_ORDER']
        self.test['DIFF_MONTH_ORDER'] = self.test['COUNT_MAX_MONTH_ORDER'] - self.test['COUNT_MIN_MONTH_ORDER']

        self.train['DIFF_QUOTA_AMT_USD'] = self.train['MAX_QUOTA_AMT_USD'] - self.train['MIN_QUOTA_AMT_USD']
        self.test['DIFF_QUOTA_AMT_USD'] = self.test['MAX_QUOTA_AMT_USD'] - self.test['MIN_QUOTA_AMT_USD']
        print("Difference Variables Created.")

        print("Size of self.train: " + str(len(self.train)))
        print("Size of self.test: " + str(len(self.test)))
        print("Size of self.train_idx: " + str(len(self.train_idx)))
        print("Size of self.test_idx: " + str(len(self.test_idx)))

    def impute_data(self):

        global MANDATORY_REPORTING_COLUMNS_LIST

        #remove reporting columns for now from self.train and self.test
        npa_train_mandatory_reporting_columns = self.train[MANDATORY_REPORTING_COLUMNS_LIST].values
        self.train = self.train.drop(MANDATORY_REPORTING_COLUMNS_LIST, axis=1)

        train_columns_list = ['BUSINESS_ID', 'CAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID'] + self.train.columns.tolist()

        npa_test_mandatory_reporting_columns = self.test[MANDATORY_REPORTING_COLUMNS_LIST].values
        self.test = self.test.drop(MANDATORY_REPORTING_COLUMNS_LIST, axis=1)

        test_columns_list = ['BUSINESS_ID', 'CAL_YEAR', 'TERM_AS_OF_DATE', 'MASTER_PARTICIPANT_ID'] + self.test.columns.tolist()

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

    def recombine_data(self):

        # Merge Complete Numeric df with complete Categorical df:
        self.tot_col = self.df.shape[1]
        one_hot_train = self.one_hot_df.iloc[self.train_idx,:]
        one_hot_test = self.one_hot_df.iloc[self.test_idx,:]
        one_hot_train.reset_index(drop = True, inplace = True)
        one_hot_test.reset_index(drop = True, inplace = True)
        
        self.train = pd.merge(self.train, one_hot_train, left_index=True, right_index=True)
        self.test = pd.merge(self.test, one_hot_test, left_index=True, right_index=True)
        
        #self.train = pd.concat([self.train, one_hot_train], axis = 1)
        #self.test = pd.concat([self.test, one_hot_test], axis = 1)

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

        global TEST_MODE_ON

        self.train.to_csv("/dbfs/FileStore/attrition_test_data/self_train_top_normalize_data.csv", index=False)

        # Split the Data/Normalize based off of percentage:
        self.tot_col = self.train.shape[1]
        X_ = self.train.values
        self.x_train = X_[:, 0:X_.shape[1] - 1]
        self.y_train = X_[:,-1]

        X_ = self.test.values
        self.x_test = X_[:, 0:X_.shape[1] - 1]
        self.y_test = X_[:,-1]

        ignore = np.arange((self.tot_col-self.num_one_hot-1), (self.tot_col-1))
        
        #add the 4 reporting columns to the list of ignored columns if using a test file
        if TEST_MODE_ON == True:
            ignore = np.concatenate((np.array([0,1,2,3]), ignore))
        
        self.x_train, self.x_test = self.normalize_x(self, self.x_train, self.x_test, ignore_cols = ignore)

        # Recreate Train and Test to query out more columns (if needed):
        columns = self.train.iloc[:,0:len(self.train.columns)-1].columns
        self.train = pd.DataFrame(self.x_train, columns = columns)
        self.test = pd.DataFrame(self.x_test, columns = columns)

        print("Normalized Data.")


    def select_features(self):

        global MANDATORY_REPORTING_COLUMNS_LIST
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

        if TEST_MODE_ON == True:
            self.features= np.concatenate((MANDATORY_REPORTING_COLUMNS_LIST, selected_feat, categ_col.columns))

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
        elif self.calendar_type == 'gregorian':
            path = '/dbfs/FileStore/pickleFiles/data_object_gregorian.obj'
            file_out = open(path, 'wb')
            dump(self, file_out)
        else:
            raise ValueError("Wrong name.")

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

        global TEST_MODE_ON

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
            
            if TEST_MODE_ON == True:
                #we need to separate out the reporting columns before doing any fitting
                npa_reporting_col_data_train = self.x_train[:, :4]
                self.x_train = self.x_train[:, 4:]
                npa_reporting_col_data_test = self.x_test[:, :4]
                self.x_test = self.x_test[:, 4:]

            self.rf.fit(self.x_train, self.y_train)
            y_test_pred = self.rf.predict(self.x_test)
            train_score = self.rf.score(self.x_train, self.y_train)
            test_score = self.rf.score(self.x_test, self.y_test)
            print("Test score: ", test_score, "Train score: ", train_score)

            #add reporting cols back if necessary
            if TEST_MODE_ON == True:
                self.x_train = np.hstack((npa_reporting_col_data_train, self.x_train))
                self.x_test = np.hstack((npa_reporting_col_data_test, self.x_test))
              
        self.nightly_file.write("Training Score: " + str(train_score) + ", Test Score: " + str(test_score)) 
        self.auc()

    def auc(self):

        global TEST_MODE_ON

        npa_reporting_col_data_test = None

        if self.model_type == 'DecisionTree':
            y_test_probs = self.dt.predict_proba(self.x_test)    
            y_test_binary = label_binarize(self.y_test, classes = [0, 1])
            y_test_probs = np.array(pd.DataFrame(y_test_probs)[1]).reshape(-1,1)
            auc = roc_auc_score(y_test_binary, y_test_probs)
        elif self.model_type == 'RandomForest':

            #strip out reporting columns, and then re-merge them at bottom
            if TEST_MODE_ON == True:
                npa_reporting_col_data_test = self.x_test[:, :4]
                self.x_test = self.x_test[:, 4:]

            y_test_probs = self.rf.predict_proba(self.x_test)    
            y_test_binary = label_binarize(self.y_test, classes = [0, 1])
            y_test_probs = np.array(pd.DataFrame(y_test_probs)[1]).reshape(-1,1)
            auc = roc_auc_score(y_test_binary, y_test_probs)

            #re-merge reporting columns
            if TEST_MODE_ON == True:
                self.x_test = np.hstack((npa_reporting_col_data_test, self.x_test))

        self.nightly_file.write(", AUC: " + str(auc))
        self.nightly_file.close()

    def predict_final(self):
        """Need to have model train on entire dataset."""
        global TEST_MODE_ON
        global MANDATORY_REPORTING_COLUMNS_LIST

        df_mandatory_reporting_columns = None

        #strip out reporting columns
        if TEST_MODE_ON == True:
            df_mandatory_reporting_columns = self.df[MANDATORY_REPORTING_COLUMNS_LIST]
            self.df = self.df.drop(MANDATORY_REPORTING_COLUMNS_LIST, axis=1)
 
        # All data needs to be trained on:
        X = self.df.drop(['TERM_NEXT_YEAR'], axis = 1).values
        Y = self.df['TERM_NEXT_YEAR'].values
        if self.model_type == 'DecisionTree':
            self.dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 100, min_samples_split = 20, ccp_alpha = 0.00001)
            self.dt.fit(X, Y)
        elif self.model_type == 'RandomForest':
            self.rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = 50, max_features = 20, min_samples_split = 10, ccp_alpha=0.00001)
            self.rf.fit(X, Y)
        
        #re-merge reporting columns
        if TEST_MODE_ON == True:
            self.df = pd.concat([df_mandatory_reporting_columns, self.df], axis=1)

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

    def __init__(self, type_model, header_path, data_path, pred_output_path, visual_output_path, model_path):
        """
        Initialized the LivePrediction Class.

        Keyword Arguments:
        type_model -- either 'fiscal' or 'gregorian'
        header_path -- the filepath to retrieve the headers of the data
        data_path -- the filepath to retreive data
        pred_output_path -- the filepath to output predictions
        visual_output_path -- the filepath to output visual weights
        model_path -- the filepath to retrieve the model 
        """
        prefix_str = ""   #change to empty string for local run.

        self.type_model = type_model
        self.data_path = data_path
        self.pred_output_path = prefix_str + pred_output_path
        self.visual_output_path = prefix_str + visual_output_path
        self.model_path = model_path

        try:
        
            self.df = pd.read_csv(self.data_path, sep='\t')

            if self.df.shape[1] == MAX_COL:
                self.df = self.df.iloc[:, 0:-2]

        except FileNotFoundError as f:
            msg = "Data Path or Header Path Incorrect."
            #logger.critical(msg)
            #logger.critical(f)
            raise FileNotFoundError(msg)
        
        try:
            self.df.reset_index(
                drop=True, inplace=True
            )
        except AssertionError as a:
            msg = "Array length and header length mismatch."
            #logger.critical(msg)
            #logger.critical(a)
            raise AssertionError(msg)

        try:
            if self.type_model == 'fiscal':
                self.data_object = load(
                    open(self.model_path + '/data_object_fiscal.obj', 'rb')
                )
                self.model = load(
                    open(self.model_path + '/rf_model_fiscal.pkl', 'rb')
                )
                if 'CAL_YEAR' in self.df.columns:
                    self.df.rename(
                        columns={"CAL_YEAR": "FISCAL_YEAR"}, inplace=True
                    )
            elif self.type_model == 'gregorian':
                self.data_object = load(
                    open(self.model_path + '/data_object_gregorian.obj', 'rb')
                )
                self.model = load(
                    open(self.model_path + '/rf_model_gregorian.pkl', 'rb')
                )
            else:
                msg = "Incorrect model flag hardcoded in script."
                #logger.critical(msg)
                raise ValueError(msg)
        except FileNotFoundError as f:
            msg = "Pickle files could not be found - either path is wrong or name of pickle files."
            #logger.critical(msg)
            #logger.critical(f)
            raise FileNotFoundError(msg)
        
        self.business_ids = None
        self.master_participant_id = None
        self.fiscal_year = None
        self.cal_year = None
        self.term_as_of_date = None
        self.results = None
        
    def save_reporting_columns(self):
        
        self.business_ids = self.df['BUSINESS_ID']
        self.master_participant_id = self.df['MASTER_PARTICIPANT_ID']
        self.term_as_of_date = self.df['TERM_AS_OF_DATE']

        if self.type_model == "gregorian":
            self.cal_year = self.df['CAL_YEAR']
        else:
            self.cal_year = self.df['FISCAL_YEAR']


    def clean_input(self):
        """ Cleans the input in the same manner as the training of the model. """

        global TEST_MODE_ON

        self.df.to_csv("/dbfs/FileStore/attrition_test_data/top_of_clean_input.csv", index=False)

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

        #write dataset to DBFS
        # if self.type_model == 'fiscal':
        #     var_df.to_csv("/dbfs/FileStore/fiscal_data_before_imputation.csv", index=False)
        # elif self.type_model == 'gregorian':
        #     var_df.to_csv("/dbfs/FileStore/gregorian_data_before_imputation.csv", index=False)

        #print("Prediction type: " + self.type_model)
        #print("Shape: " + str(var_df.shape))
        #print("Column List: " + str(var_df.columns.tolist()))
        #print('--------')
        
        #fill in blank values to avoid generating a ValueError
        #var_df = var_df.fillna(0)

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
            #logger.critical(msg)
            #logger.critical(v)
            raise ValueError(msg)

        # if self.type_model == 'fiscal':
        #     var_df.to_csv("/dbfs/FileStore/fiscal_data_after_imputation.csv", index=False)
        # elif self.type_model == 'gregorian':
        #     var_df.to_csv("/dbfs/FileStore/gregorian_data_after_imputation.csv", index=False)

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

        reporting_column_list = MANDATORY_REPORTING_COLUMNS_LIST.copy()

        """ Predict the Attrition likelihood for input data and outputs the results as a tsv. """
        try:
            
            if self.type_model == "fiscal":
                cal_year_idx = reporting_column_list.index("CAL_YEAR")
                reporting_column_list[cal_year_idx] = "FISCAL_YEAR"

            df_mandatory_reporting_columns = self.df[reporting_column_list]
            self.df = self.df.drop(reporting_column_list, axis=1)

            rf_model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = 50, max_features = 20, min_samples_split = 10, ccp_alpha=0.00001)

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

            train_score = rf_model.score(x_train, y_train)
            test_score = rf_model.score(x_test, y_test)
            print("Test score: ", test_score, "Train score: ", train_score)

            #generate preds and probs for self.df
            #first, extract target
            self.df = self.df.drop(['TERM_NEXT_YEAR'], axis=1)

            probs = rf_model.predict_proba(self.df.values)
          
            #print('Executed predict_proba successfully')


        except ValueError as v:
            exception_str = repr(v)
            msg = "Array passed into model is of incorrect length."
            print(exception_str)
            #logger.critical(msg)
            #logger.critical(v)
            raise ValueError(msg)
        except:
            msg = "Unspecified Error when making prediction."
            #logger.critical(msg)
            raise ValueError(msg)

        if self.type_model == 'fiscal':
            
            print("in fiscal if branch..")

            print("self.business_ids.shape = " + str(self.business_ids.shape))
            print("self.master_participant_id.shape = " + str(self.master_participant_id.shape))

            d = {
                'BUSINESS_ID': self.business_ids, 'MASTER_PARTICIPANT_ID': self.master_participant_id,
                'FISCAL_YEAR': self.fiscal_year, 'PRED_TERM_PROB': probs[:, 1]
                }
            
            try:
                
                self.results = pd.DataFrame(d)

                if TEST_MODE_ON == True:
                    self.results.to_csv(self.pred_output_path + "results_test_fiscal.csv", index=False)
                else:

                    df_spark = spark.createDataFrame(self.results)

                    df_spark.write \
                        .format("snowflake") \
                        .mode("overwrite") \
                        .options(**options) \
                        .option("dbtable", "TURNOVER_PRED_FISCAL") \
                        .save()
                
                #get today's date and time as a string
                current_datetime_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

                #write data to Databricks as well, whether we are using a test file or not
                self.results.to_csv(
                    self.pred_output_path + '/part_fiscal_pred_' + current_datetime_str + '.tsv',
                    header = None, index=False, sep='\t'
                )
                
            except FileNotFoundError as f:
                msg = "Fiscal output filepath not found when outputting."
                #logger.critical(msg)
                #logger.critical(f)
                raise FileNotFoundError(msg)

        elif self.type_model == 'gregorian':

            print("in gregorian if branch..")

            print("self.business_ids.shape = " + str(self.business_ids.shape))
            print("self.master_participant_id.shape = " + str(self.master_participant_id.shape))

            d = {
                'BUSINESS_ID': self.business_ids, 'MASTER_PARTICIPANT_ID': self.master_participant_id,
                'CAL_YEAR': self.cal_year, 'PRED_TERM_PROB': probs[:, 1]
                }

            try:

                self.results = pd.DataFrame(d)
                
                if TEST_MODE_ON == True:
                    self.results.to_csv(self.pred_output_path + "results_test_gregorian.csv", index=False)
                else:


                    df_spark = spark.createDataFrame(self.results)

                    df_spark.write \
                        .format("snowflake") \
                        .mode("overwrite") \
                        .options(**options) \
                        .option("dbtable", "TURNOVER_PRED_GREGORIAN") \
                        .save()
                
                #get today's date and time as a string
                current_datetime_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

                #write data to Databricks as well
                self.results.to_csv(
                   self.pred_output_path + '/part_gregorian_pred_' + current_datetime_str + '.tsv',
                   header = None, index=False, sep='\t'
                )
               
            except FileNotFoundError as f:
                msg = "Gregorian output filepath not found when outputting."
                #logger.critical(msg)
                #logger.critical(f)
                raise FileNotFoundError(msg)
        print("Predictions outputted successfully.")
        print("SUCCESS.")

    def explain_pred(self):
        """ 
        Explain the predictions for the input data, outputted as WEIGHTS_ATTRITION.tsv 
        DO NOT RUN - Need to import SHAP
        """
        global TEST_MODE_ON

        print("Explainer started.")

        print("self.results column list = " + str(self.results.columns.tolist()))

        shap_exp = shap.TreeExplainer(
            self.model, background=self.data_object.x_train
        )

        print("Shap Explainer Created.")
        shap_values_test = shap_exp.shap_values(
            self.df.values, approximate=True,
            check_additivity=False
        )

        print("Shap Values Created.")
        shap_arr = np.zeros(
            self.df.shape
        )
        try:
            for row in range(len(self.df)):
                vals = shap_values_test[1][row]
                shap_arr[row] = vals
        except IndexError as i:
            msg = "No shap values were created from the TreeExplainer."
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

        #if self.type_model == 'gregorian':
        #    del self.results["COMP_AVG_MONTH_PAYEECOUNT"]

        self.results.to_csv(
            self.visual_output_path + '/part_visual_' + self.type_model + '.tsv',
            header=None, index=False, sep='\t'
        )

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

# COMMAND ----------

#read the TRIGGER_ATTRITION table
df_trigger_attrition = spark.read \
                    .format("snowflake") \
                    .options(**options) \
                    .option("dbtable","INSIGHTS_PARAMETER") \
                    .load().toPandas()

#display(df_trigger_attrition)

#if the flag is False,then we don't execute attrition
trigger_attrition_flag = df_trigger_attrition.loc[0, 'VALUE']

if trigger_attrition_flag == 'TRUE' or TEST_MODE_ON == True:

    #create cleaning object for gregorian predictions - but only for real files
    gregorian_data = CleanData('gregorian', test_mode=TEST_MODE_ON)
    rfModelGregorian = RandomForest(gregorian_data, 'RandomForest')
    rfModelGregorian.predict()
    rfModelGregorian.predict_final()
    rfModelGregorian.save_model()
    gregorian_data.save_data_object()

    print("Finished all gregorian stuff, on to fiscal stuff....")

    #create cleaning object for fiscal predictions
    fiscal_data = CleanData('fiscal', test_mode=TEST_MODE_ON)
    rfModelFiscal = RandomForest(fiscal_data, 'RandomForest')
    rfModelFiscal.predict()
    rfModelFiscal.predict_final()
    rfModelFiscal.save_model()
    fiscal_data.save_data_object()

    print("all done with cleaning!")
    print("let's start making predictions....")
    print('----------')

    #execute both fiscal and gregorian attrition

    fiscal_args = ["fiscal", fiscal_header_path_cl, fiscal_file_path_cl, fiscal_output_path, fiscal_vis_output_path, model_input_cl]
    gregorian_args = ["gregorian", gregorian_header_path_cl, gregorian_file_path_cl, gregorian_output_path, gregorian_vis_output_path, model_input_cl]

    for input_lst in [gregorian_args, fiscal_args]:
        batch_pred = LivePrediction(
            input_lst[0], input_lst[1], input_lst[2],
            input_lst[3], input_lst[4], input_lst[5]
        )

        #batch_pred.clean_input()
        batch_pred.save_reporting_columns()
        batch_pred.predict()
        batch_pred.explain_pred()
    
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


