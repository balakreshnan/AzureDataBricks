// Databricks notebook source
// MAGIC %sh 
// MAGIC telnet 40.123.38.174 9092

// COMMAND ----------



// COMMAND ----------

val df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "40.123.38.174:9092").option("subscribe", "sometopic").option("startingOffsets", "earliest").load()

// COMMAND ----------

df.printSchema

// COMMAND ----------

import org.apache.spark.sql.functions._

var streamingSelectDF = df.select(get_json_object(($"value").cast("string"), "$.deviceId").alias("deviceid"),
get_json_object(($"value").cast("string"), "$.temperature").alias("temperature")
,get_json_object(($"value").cast("string"), "$.humidity").alias("humidity")
,get_json_object(($"value").cast("string"), "$.sensortime").alias("sensortime")
,date_format(get_json_object(($"value").cast("string"), "$.sensortime"), "dd.MM.yyyy").alias("day"))

// COMMAND ----------

import org.apache.spark.sql.streaming.ProcessingTime

val query = streamingSelectDF.writeStream.format("parquet").option("path","/mnt/data/device").option("checkpointLocation","/mnt/data/check").partitionBy("deviceId","day").trigger(ProcessingTime("10 seconds")).start()

// COMMAND ----------

val streamData = spark.read.parquet("/mnt/data/device")
streamData.show()

// COMMAND ----------

// MAGIC %sql DROP TABLE test_dev

// COMMAND ----------

// MAGIC %sql CREATE EXTERNAL TABLE test_dev (temperature string,humidity string,sennsortime string) 
// MAGIC PARTITIONED BY(deviceId string, day string)
// MAGIC STORED AS PARQUET
// MAGIC LOCATION "/mnt/data/device"

// COMMAND ----------

// MAGIC %sql select * from test_dev

// COMMAND ----------

display(df)

// COMMAND ----------

df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

// COMMAND ----------

schema = StructType().add("deviceId", StringType()).add("temperature", StringType()).add("humidity", StringType()).add("sensortime", StringType())
df.select(col("key").cast("string"),from_json(col("value").cast("string"), schema))