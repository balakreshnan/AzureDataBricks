# Databricks notebook source
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
import time

# COMMAND ----------

storagename = "dewsa"
storagekey = "TDjjpKGq75D4t0fy4D5fNTJrnzlrtCKxB9MIipsfQvLWsR3mScAswOZTDOMi0JSXi1fb2ga5ly4rUAQ+WUvETQ=="

storageURL = "wasbs://cat@dewsa.blob.core.windows.net/"


# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key.dewsa.blob.core.windows.net",
  "TDjjpKGq75D4t0fy4D5fNTJrnzlrtCKxB9MIipsfQvLWsR3mScAswOZTDOMi0JSXi1fb2ga5ly4rUAQ+WUvETQ==")

# COMMAND ----------

val df = spark.read.parquet("wasbs://cat@dewsa.blob.core.windows.net/")
dbutils.fs.ls("wasbs://cat@dewsa.blob.core.windows.net/")

# COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://cat@dewsa.blob.core.windows.net/",
  mount_point = "/mnt/data",
  extra_configs = {"fs.azure.account.key.dewsa.blob.core.windows.net": "TDjjpKGq75D4t0fy4D5fNTJrnzlrtCKxB9MIipsfQvLWsR3mScAswOZTDOMi0JSXi1fb2ga5ly4rUAQ+WUvETQ=="})

# COMMAND ----------

# MAGIC %fs ls /mnt/data

# COMMAND ----------

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}

# COMMAND ----------

csvFile = "dbfs:/mnt/data/344_blend_pl_sos_impute_sample.csv"
df = spark.read_csv(csvFile)
df.show()


# COMMAND ----------

csvFile = "dbfs:/mnt/data/344_blend_pl_sos_impute_sample.csv"
sosDF = (spark.read                   
  .option('header', 'true') 
  .schema('inferSchema','true') 
  .csv(csvFile)
)
sosDF.createOrReplaceTempView("sos")

# COMMAND ----------

df = spark.read.csv('dbfs:/mnt/data/344_blend_pl_sos_impute_sample.csv', header='true', inferSchema = 'true')

# COMMAND ----------

display(df.limit(5))

# COMMAND ----------

df.createOrReplaceTempView("sos")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sos limit 10

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from sos

# COMMAND ----------

df.count()

# COMMAND ----------

