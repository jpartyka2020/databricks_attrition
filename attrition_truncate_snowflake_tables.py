# Databricks notebook source
# MAGIC %scala
# MAGIC
# MAGIC import net.snowflake.spark.snowflake.Utils
# MAGIC import org.apache.spark.sql.{SparkSession, DataFrame}
# MAGIC import org.apache.spark.sql.functions.col
# MAGIC import scala.sys.exit
# MAGIC import scala.io
# MAGIC
# MAGIC // Snowflake credentials
# MAGIC val user = "app_datascience"
# MAGIC val password = "Xactly123"
# MAGIC
# MAGIC val options = Map (
# MAGIC   "sfUrl" -> "https://xactly-xactly_engg_datalake_aws.snowflakecomputing.com/",
# MAGIC   "sfUser" -> user,
# MAGIC   "sfPassword" -> password,
# MAGIC   "sfDatabase" -> "XTLY_ENGG",
# MAGIC   "sfSchema" -> "INSIGHTS",
# MAGIC   "sfWarehouse" -> "DIS_LOAD_WH",
# MAGIC   "truncate_table" -> "ON",
# MAGIC   "usestagingtable" -> "OFF"
# MAGIC )
# MAGIC
# MAGIC //we will put one line for each table that we want to truncate
# MAGIC //TURNOVER_PRED_GREGORIAN
# MAGIC //TURNOVER_PRED_FISCAL
# MAGIC //VISUALISATION_FISCAL
# MAGIC //VISUALIZATION_GREGORIAN
# MAGIC //practice: TURNOVER_PRED_FISCAL_TEMP1
# MAGIC
# MAGIC Utils.runQuery(options, "TRUNCATE TABLE TURNOVER_PRED_GREGORIAN")
# MAGIC Utils.runQuery(options, "TRUNCATE TABLE TURNOVER_PRED_FISCAL")
# MAGIC Utils.runQuery(options, "TRUNCATE TABLE VISUALISATION_GREGORIAN")
# MAGIC Utils.runQuery(options, "TRUNCATE TABLE VISUALISATION_FISCAL")
# MAGIC
# MAGIC //we also want to effectively truncate the INSIGHTS_PARAMETER table by using a DELETE query
# MAGIC val options_ip = Map (
# MAGIC   "sfUrl" -> "https://xactly-xactly_engg_datalake_aws.snowflakecomputing.com/",
# MAGIC   "sfUser" -> user,
# MAGIC   "sfPassword" -> password,
# MAGIC   "sfDatabase" -> "XTLY_ENGG",
# MAGIC   "sfSchema" -> "INSIGHTS",
# MAGIC   "sfWarehouse" -> "DIS_LOAD_WH"
# MAGIC )
# MAGIC
# MAGIC Utils.runQuery(options_ip, "DELETE FROM INSIGHTS_PARAMETER WHERE name='TRIGGER_ATTRITION'")
# MAGIC
# MAGIC //if we get here, then the job was successful
# MAGIC dbutils.notebook.exit("success")
