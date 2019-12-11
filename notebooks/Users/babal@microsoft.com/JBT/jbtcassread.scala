// Databricks notebook source
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType,LongType,FloatType,DoubleType, TimestampType}
import org.apache.spark.sql.cassandra._

//datastax Spark connector
import com.datastax.spark.connector._
import com.datastax.spark.connector.cql.CassandraConnector

//CosmosDB library for multiple retry
import com.microsoft.azure.cosmosdb.cassandra

// COMMAND ----------

// Specify connection factory for Cassandra
spark.conf.set("spark.cassandra.connection.factory", "com.microsoft.azure.cosmosdb.cassandra.CosmosDbConnectionFactory")


// COMMAND ----------

// Parallelism and throughput configs
spark.conf.set("spark.cassandra.output.batch.size.rows", "1")
spark.conf.set("spark.cassandra.connection.connections_per_executor_max", "10")
spark.conf.set("spark.cassandra.output.concurrent.writes", "10")
spark.conf.set("spark.cassandra.concurrent.reads", "10")
spark.conf.set("spark.cassandra.output.batch.grouping.buffer.size", "1000")
spark.conf.set("spark.cassandra.connection.keep_alive_ms", "60000000") //Increase this number as needed
//spark.conf.set("spark.cassandra.output.consistency.level","ALL")//Write consistency = Strong
//spark.conf.set("spark.cassandra.input.consistency.level","ALL")//Read consistency = Strong

// COMMAND ----------

spark.conf.set("spark.cassandra.connection.host", "jbtiotcassandra.cassandra.cosmos.azure.com")
spark.conf.set("spark.cassandra.connection.port", "10350")
spark.conf.set("spark.cassandra.connection.ssl.enabled", "true")
spark.conf.set("spark.cassandra.auth.username", "jbtiotcassandra")
spark.conf.set("spark.cassandra.auth.password", "4h1l8iCpUHIZxrnwjgLBydslWQOJ2HVa1y4dFbG2QmoZpUk8eAgiT7CRrsEWj6x4N7YrOgGh6Y77dfUguqcDfg==")

// COMMAND ----------

val readBasketDF = spark
  .read
  .format("org.apache.spark.sql.cassandra")
  .options(Map( "table" -> "basket", "keyspace" -> "jbtiot"))
  .load

readBasketDF.explain
readBasketDF.show

// COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://incoming@jbtiotstore.blob.core.windows.net/<directory-name>",
  mountPoint = "/mnt/jbtincoming",
  extraConfigs = Map("<conf-key>" ->"n0Uk6apI8w1pvjR16x6sQN6aKrijYVaG4ZwAvpU4legWHqAf3qrJimnEuYHY2KsOvRbXB40DPBAv3A/JbwwGrw=="))

// COMMAND ----------

spark.conf.set(
  "fs.azure.account.key.jbtiotstore.blob.core.windows.net",
  "n0Uk6apI8w1pvjR16x6sQN6aKrijYVaG4ZwAvpU4legWHqAf3qrJimnEuYHY2KsOvRbXB40DPBAv3A/JbwwGrw==")

// COMMAND ----------

val df = spark.read.option("header", "true").option("inferSchema","true").csv("wasbs://incoming@jbtiotstore.blob.core.windows.net/BTS_vwBasket.csv")

// COMMAND ----------

display(df)

// COMMAND ----------

df.count

// COMMAND ----------

df.write
  .mode("append")
  .format("org.apache.spark.sql.cassandra")
  .options(Map( "table" -> "basket", "keyspace" -> "jbtiot"))
  .save()

// COMMAND ----------

// MAGIC %sql
// MAGIC select UUID()

// COMMAND ----------

//import sqlContext.implicits._

import org.apache.spark.sql.ForeachWriter
 import org.apache.spark.sql.Row
 import java.io.File
 import org.apache.commons.io.FileUtils
 import java.util.UUID.randomUUID
 import java.nio.charset.StandardCharsets

//display(java.util.UUID.randomUUID().toString)
printf("Create New UUID: %s\n",java.util.UUID.randomUUID().toString)

// COMMAND ----------

//val sc: SparkContext = ...
//val sqlContext = new SQLContext(sc)
import sqlContext.implicits._
import java.util.UUID.randomUUID

val generateUUID = udf(() => java.util.UUID.randomUUID().toString)
val df1 = Seq(("id1", 1), ("id2", 4), ("id3", 5)).toDF("id", "value")
val df2 = df1.withColumn("UUID", generateUUID())
df1.show()
df2.show()

// COMMAND ----------

val dfwithUUID = df.withColumn("uuidcol",generateUUID())

// COMMAND ----------

display(dfwithUUID)

// COMMAND ----------

dfwithUUID.schema

// COMMAND ----------

dfwithUUID.write
  .mode("append")
  .format("org.apache.spark.sql.cassandra")
  .options(Map( "table" -> "basket", "keyspace" -> "jbtiot"))
  .save()

// COMMAND ----------

val df3 = dfwithUUID.filter($"serial_number" === "HoN6kQNFBvIeJJgjSVNYzySBHO4")

// COMMAND ----------

display(df3)

// COMMAND ----------

val readBasketDF = spark
  .read
  .format("org.apache.spark.sql.cassandra")
  .options(Map( "table" -> "basket", "keyspace" -> "jbtiot"))
  .load

//readBasketDF.explain
//readBasketDF.show

// COMMAND ----------

display(readBasketDF)

// COMMAND ----------

val df3 = readBasketDF.filter($"uuidcol" === "b2b50ce2-061a-4729-a5ca-c175ba1da65c" || $"serial_number" === "HoN6kQNFBvIeJJgjSVNYzySBHO4")

// COMMAND ----------

display(df3)

// COMMAND ----------

readBasketDF.createOrReplaceTempView("basketsql")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from basketsql where uuidcol = 'b2b50ce2-061a-4729-a5ca-c175ba1da65c' --and serial_number = 'HoN6kQNFBvIeJJgjSVNYzySBHO4'

// COMMAND ----------

val batchesdf = spark.read.option("header", "true").option("inferSchema","true").csv("wasbs://incoming@jbtiotstore.blob.core.windows.net/BTS_vwBatches.csv")

// COMMAND ----------

batchesdf.write
  .mode("append")
  .format("org.apache.spark.sql.cassandra")
  .options(Map( "table" -> "batches", "keyspace" -> "jbtiot"))
  .save()

// COMMAND ----------

val opsdf = spark.read.option("header", "true").option("inferSchema","true").csv("wasbs://incoming@jbtiotstore.blob.core.windows.net/BTS_vwiOPS1.csv")

// COMMAND ----------

opsdf.write
  .mode("append")
  .format("org.apache.spark.sql.cassandra")
  .options(Map( "table" -> "ops", "keyspace" -> "jbtiot"))
  .save()

// COMMAND ----------

val productiondf = spark.read.option("header", "true").option("inferSchema","true").csv("wasbs://incoming@jbtiotstore.blob.core.windows.net/BTS_vwProduction2.csv")

// COMMAND ----------

productiondf.write
  .mode("append")
  .format("org.apache.spark.sql.cassandra")
  .options(Map( "table" -> "production", "keyspace" -> "jbtiot"))
  .save()

// COMMAND ----------

val skudf = spark.read.option("header", "true").option("inferSchema","true").csv("wasbs://incoming@jbtiotstore.blob.core.windows.net/BTS_vwSKU.csv")

// COMMAND ----------

skudf.write
  .mode("append")
  .format("org.apache.spark.sql.cassandra")
  .options(Map( "table" -> "sku", "keyspace" -> "jbtiot"))
  .save()