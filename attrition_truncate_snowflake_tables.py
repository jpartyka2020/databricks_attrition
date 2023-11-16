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
# MAGIC Utils.runQuery(options, "TRUNCATE TABLE TURNOVER_PRED_FISCAL_TEMP1")
# MAGIC // Utils.runQuery(options, "TRUNCATE TABLE TURNOVER_PRED_GREGORIAN")
# MAGIC // Utils.runQuery(options, "TRUNCATE TABLE TURNOVER_PRED_FISCAL")
# MAGIC // Utils.runQuery(options, "TRUNCATE TABLE VISUALIZATION_GREGORIAN")
# MAGIC // Utils.runQuery(options, "TRUNCATE TABLE VISUALIZATION_FISCAL")
# MAGIC
# MAGIC //if we get here, then the job was successful
# MAGIC dbutils.notebook.exit("success")
