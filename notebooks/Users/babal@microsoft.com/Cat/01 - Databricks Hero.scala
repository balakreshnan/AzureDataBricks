// Databricks notebook source
// MAGIC %md
// MAGIC # Markdown for titles
// MAGIC ## Documentation
// MAGIC ### and more...

// COMMAND ----------

// MAGIC %scala
// MAGIC 1 + 1

// COMMAND ----------

// MAGIC %python
// MAGIC 1 + 1

// COMMAND ----------

// MAGIC %md
// MAGIC ### Mount a Blob storage account and verify its contents

// COMMAND ----------

//dbutils.fs.unmount("/mnt/training-msft")
dbutils.fs.mount(
    source = "wasbs://training-primary@databrickstraining.blob.core.windows.net/",
    mountPoint = "/mnt/training-msft",
    extraConfigs = Map("fs.azure.account.key.databrickstraining.blob.core.windows.net" ->
                       "BXOG8lPEcgSjjlmsOgoPdVCpPDM/RwfN1QTrlXEX3oq0sSbNZmNPyE8By/7l9J1Z7SVa8hsKHc48qBY1tA/mgQ=="))

// COMMAND ----------

// MAGIC %fs ls /mnt/training-msft

// COMMAND ----------

// MAGIC %md
// MAGIC ## Mount Azure Data Lake Store and verify its contents

// COMMAND ----------

//dbutils.fs.unmount("/mnt/training-adl")

val directoryId = "72f988bf-86f1-41af-91ab-2d7cd011db47"

dbutils.fs.mount(
    mountPoint = "/mnt/training-adl",
    source = "adl://bluewateradl.azuredatalakestore.net",
    extraConfigs = Map(
      "dfs.adls.oauth2.access.token.provider.type" -> "ClientCredential",
      "dfs.adls.oauth2.client.id" -> "f92a2e15-0b57-48b3-b32a-1e8d6d7055c8",
      "dfs.adls.oauth2.credential" -> "BAzcNzjMf50zrbV8ll5+IQaibNH3wIMb7kL+4AIGLYA=",
      "dfs.adls.oauth2.refresh.url" -> s"https://login.microsoftonline.com/$directoryId/oauth2/token"
    )
  )

// COMMAND ----------

// MAGIC %fs ls /mnt/training-adl

// COMMAND ----------

// MAGIC %md
// MAGIC ## 1. Read a file from blob store using Python
// MAGIC ## 2. Save as temp table and use Spark SQL
// MAGIC ## 3. Write to ADLS using Scala

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql.types import *
// MAGIC 
// MAGIC csvSchema = StructType([
// MAGIC   StructField("product_id", LongType(), True),
// MAGIC   StructField("category", StringType(), True),
// MAGIC   StructField("brand", StringType(), True),
// MAGIC   StructField("model", StringType(), True),
// MAGIC   StructField("price", DoubleType(), True),
// MAGIC   StructField("processor", StringType(), True),
// MAGIC   StructField("size", StringType(), True),
// MAGIC   StructField("display", StringType(), True)
// MAGIC  ])
// MAGIC 
// MAGIC csvFile = "dbfs:/mnt/training-msft/initech/Product.csv"
// MAGIC 
// MAGIC productDF = (spark.read                   
// MAGIC   .option('header', 'true') 
// MAGIC   .schema(csvSchema) 
// MAGIC   .csv(csvFile)
// MAGIC )
// MAGIC 
// MAGIC productDF.createOrReplaceTempView("products")

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select * from products limit 10;

// COMMAND ----------

val df = spark.sql("select * from products")
df.write.mode("overwrite").parquet("dbfs:/mnt/training-adl/initech/Products.parquet") 

// COMMAND ----------

// MAGIC %fs ls /mnt/training-adl/initech

// COMMAND ----------

