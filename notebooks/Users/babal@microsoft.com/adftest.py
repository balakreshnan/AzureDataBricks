# Databricks notebook source
# Creating widgets for leveraging parameters, and printing the parameters

dbutils.widgets.text("input", "","")
dbutils.widgets.get("input")
y = getArgument("input")
print("Param -\'input':")
print(y)

# COMMAND ----------

#dbutils.fs.mount(
#  source = "wasbs://source@dewsa.blob.core.windows.net/",
#  mount_point = "/mnt/source",
#  extra_configs = {"fs.azure.account.key.dewsa.blob.core.windows.net": "TDjjpKGq75D4t0fy4D5fNTJrnzlrtCKxB9MIipsfQvLWsR3mScAswOZTDOMi0JSXi1fb2ga5ly4rUAQ+WUvETQ=="}) 


# COMMAND ----------

dbutils.fs.ls("/mnt/source")

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/source/all_downtime_line7_jan01_to_mar10.csv")

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %scala
# MAGIC Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver")

# COMMAND ----------

SQLDBUsername = "sqladmin"
SQLDBPassword = "Azure!2345"
jdbcUrl = "jdbc:sqlserver://dewdbsvr.database.windows.net:1433;database=dewdb"
connectionProperties = {
  "user" : SQLDBUsername,
  "password" : SQLDBPassword,
  "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}
jcifaults = spark.read.jdbc(url=jdbcUrl, table="dbo.jcifaults", properties=connectionProperties)
display(jcifaults)

# COMMAND ----------

df.write.mode('overwrite').jdbc(url=jdbcUrl, table="dbo.jcifaultsspark", properties=connectionProperties)