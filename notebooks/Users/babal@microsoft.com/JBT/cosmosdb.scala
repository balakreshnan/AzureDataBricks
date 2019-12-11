// Databricks notebook source
import org.joda.time._
import org.joda.time.format._

import com.microsoft.azure.cosmosdb.spark.schema._
import com.microsoft.azure.cosmosdb.spark.CosmosDBSpark
import com.microsoft.azure.cosmosdb.spark.config.Config

import org.apache.spark.sql.functions._

// COMMAND ----------

// Configure the connection to your collection in Cosmos DB.
// Please refer to https://github.com/Azure/azure-cosmosdb-spark/wiki/Configuration-references
// for the description of the available configurations.
val configMap = Map(
  "Endpoint" -> "https://jbttssql.documents.azure.com:443/",
  "Masterkey" -> "0UfFb35wFgR28mXB9NJuFk22gqLAsHMqoy6pbLnC7KXoRfHKEBhRWJQUZtxMP0TKMGQf97WTt9mFYDzCRz8ooQ==",
  "Database" -> "jbtiot",
  "Collection" -> "Basket",
  "preferredRegions" -> "East US 2")
val config = Config(configMap)

// COMMAND ----------

// Generate a simple dataset containing five values and
// write the dataset to Cosmos DB.
val df = spark.range(5).select(col("id").cast("string").as("value"))
CosmosDBSpark.save(df, config)

// COMMAND ----------

val df1 = spark.read.cosmosDB(config)
df1.count()

// COMMAND ----------

display(df1)

// COMMAND ----------

df1.schema

// COMMAND ----------

val configMap = Map(
  "Endpoint" -> "https://jbttssql.documents.azure.com:443/",
  "Masterkey" -> "0UfFb35wFgR28mXB9NJuFk22gqLAsHMqoy6pbLnC7KXoRfHKEBhRWJQUZtxMP0TKMGQf97WTt9mFYDzCRz8ooQ==",
  "Database" -> "jbtiot",
  "Collection" -> "Batches",
  "preferredRegions" -> "East US 2")
val config = Config(configMap)

// COMMAND ----------

val batches = spark.read.cosmosDB(config)
batches.count()

// COMMAND ----------

display(batches)

// COMMAND ----------

val configMap = Map(
  "Endpoint" -> "https://jbttssql.documents.azure.com:443/",
  "Masterkey" -> "0UfFb35wFgR28mXB9NJuFk22gqLAsHMqoy6pbLnC7KXoRfHKEBhRWJQUZtxMP0TKMGQf97WTt9mFYDzCRz8ooQ==",
  "Database" -> "jbtiot",
  "Collection" -> "sampledatapla",
  "preferredRegions" -> "East US 2")
val config = Config(configMap)

// COMMAND ----------

val sampledataplc = spark.read.cosmosDB(config)
sampledataplc.count()

// COMMAND ----------

display(sampledataplc)