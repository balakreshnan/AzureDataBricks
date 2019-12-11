// Databricks notebook source
val df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "52.232.160.222:9092").option("subscribe", "sometopic").option("startingOffsets", "earliest").load()

// COMMAND ----------

// MAGIC %sh
// MAGIC nc -vz 52.232.160.222 9092

// COMMAND ----------

// MAGIC %sh telnet 52.232.160.222 9092

// COMMAND ----------

display(df)

// COMMAND ----------

// MAGIC %sh
// MAGIC nc -vz 40.123.26.57 9092

// COMMAND ----------

val df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "40.123.26.57:9092").option("subscribe", "sometopic").option("startingOffsets", "earliest").load()

// COMMAND ----------

display(df)