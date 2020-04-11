# Databricks notebook source
# MAGIC %scala
# MAGIC Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver")
# MAGIC Class.forName("com.databricks.spark.sqldw.DefaultSource")

# COMMAND ----------

#https://docs.azuredatabricks.net/spark/latest/data-sources/sql-databases.html

# COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://jci@dewsa.blob.core.windows.net/",
  mount_point = "/mnt/jci",
  extra_configs = {"fs.azure.account.key.dewsa.blob.core.windows.net": "TDjjpKGq75D4t0fy4D5fNTJrnzlrtCKxB9MIipsfQvLWsR3mScAswOZTDOMi0JSXi1fb2ga5ly4rUAQ+WUvETQ=="})

# COMMAND ----------

# MAGIC %fs ls /mnt/jci

# COMMAND ----------

display(dbutils.fs.ls("file:/mnt/jci"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/jci"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/jci"))

# COMMAND ----------

username = "sqladmin"
password = "Azure!2345"
jdbcHostname = "dewdbsvr.database.windows.net"
jdbcDatabase = "dewdb"
jdbcPort = 1433
jdbcUrl = "jdbc:sqlserver://{0}:{1};database={2};user={3};password={4}".format(jdbcHostname, jdbcPort, jdbcDatabase, username, password)

# COMMAND ----------

jdbcUrl = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname, jdbcPort, jdbcDatabase)
connectionProperties = {
  "user" : username,
  "password" : password,
  "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# COMMAND ----------

pushdown_query = "(select * from jcifaults) as faults"
df = spark.read.jdbc(url=jdbcUrl, table=pushdown_query, properties=connectionProperties)
display(df)

# COMMAND ----------

df1 = spark.read.jdbc(url=jdbcUrl, table="jcifaults", properties=connectionProperties, column="StartTime", lowerBound=1, upperBound=100000, numPartitions=100)
display(df1)

# COMMAND ----------

display(df)

# COMMAND ----------

# import pyspark class Row from module sql
from pyspark.sql import *
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

df.registerTempTable("faults")
display(sql("select * from faults"))

# COMMAND ----------

display(sql("select max(Line) as Line, max(Unit) as Unit, max(FaultDescription) as FaultDescription, ReasonLevel1, ReasonLevel2, ReasonLevel3, ReasonLevel4, sum(Duration) as Duration, sum(Uptime) as Uptime from faults group by ReasonLevel1, ReasonLevel2, ReasonLevel3, ReasonLevel4"))

# COMMAND ----------

df_2 = spark.sql("select max(Line) as Line, max(Unit) as Unit, max(FaultDescription) as FaultDescription, ReasonLevel1, ReasonLevel2, ReasonLevel3, ReasonLevel4, sum(Duration) as Duration, sum(Uptime) as Uptime from faults group by ReasonLevel1, ReasonLevel2, ReasonLevel3, ReasonLevel4")

# COMMAND ----------

display(df_2)

# COMMAND ----------

display(df_2)

# COMMAND ----------

display(df_2)

# COMMAND ----------

display(df_2)

# COMMAND ----------

display(df_2)

# COMMAND ----------

display(df_2)

# COMMAND ----------

#only to clean up
sqlContext.clearCache()
sqlContext.cacheTable("faults")
sqlContext.uncacheTable("faults")

# COMMAND ----------

#this is for SQl DW connectivity
#https://docs.databricks.com/spark/latest/data-sources/azure/sql-data-warehouse.html

# COMMAND ----------

# Set up the Blob Storage account access key in the notebook session conf.
spark.conf.set(
  "fs.azure.account.key.dewsa.blob.core.windows.net",
  "TDjjpKGq75D4t0fy4D5fNTJrnzlrtCKxB9MIipsfQvLWsR3mScAswOZTDOMi0JSXi1fb2ga5ly4rUAQ+WUvETQ==")

# COMMAND ----------

dwusername = "VISADMIN@visprddb"
dwpassword = "KohlerVi$01"
dwjdbcHostname = "visprddb.database.windows.net"
dwjdbcDatabase = "VIS_PRD_SQLDW"
dwjdbcPort = 1433
dwjdbcUrl = "jdbc:sqlserver://visprddb.database.windows.net:1433;database=VIS_PRD_SQLDW;user=VISADMIN@visprddb;password=KohlerVi$01;encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"

# COMMAND ----------

# Get some data from a SQL DW table.
dfyield = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", dwjdbcUrl) \
  .option("tempdir", "wasbs://tempdata@dewsa.blob.core.windows.net/temp") \
  .option("forward_spark_azure_storage_credentials", "true") \
  .option("dbtable", "Yield") \
  .load() 

# COMMAND ----------

display(dfyield)

# COMMAND ----------

