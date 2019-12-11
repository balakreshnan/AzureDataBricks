// Databricks notebook source
// MAGIC %md ## Spark-Tensorflow-Connector Example
// MAGIC 
// MAGIC This notebook (adapted from the spark-tensorflow-connector
// MAGIC [usage examples](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector#usage-examples)) demonstrates exporting Spark DataFrames to TFRecords and loading the exported TFRecords back into DataFrames.

// COMMAND ----------

import org.apache.commons.io.FileUtils
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._

// COMMAND ----------

// MAGIC %md 
// MAGIC #### Create DataFrame
// MAGIC Construct a DataFrame with columns of various (int, long, float, array, string) types

// COMMAND ----------

// Declare DataFrame data
val testRows: Array[Row] = Array(
  new GenericRow(Array[Any](11, 1, 23L, 10.0F, 14.0, List(1.0, 2.0), "r1")),
  new GenericRow(Array[Any](21, 2, 24L, 12.0F, 15.0, List(2.0, 2.0), "r2"))
)

// DataFrame schema
val schema = StructType(List(StructField("id", IntegerType), 
                             StructField("IntegerTypeLabel", IntegerType),
                             StructField("LongTypeLabel", LongType),
                             StructField("FloatTypeLabel", FloatType),
                             StructField("DoubleTypeLabel", DoubleType),
                             StructField("VectorLabel", ArrayType(DoubleType, true)),
                             StructField("name", StringType)))
// Create DataFrame
val rdd = spark.sparkContext.parallelize(testRows)
val df: DataFrame = spark.createDataFrame(rdd, schema)

// COMMAND ----------

df.show()

// COMMAND ----------

// MAGIC %md #### Export DataFrame to TFRecords
// MAGIC 
// MAGIC WARNING: The command below will overwrite existing data

// COMMAND ----------

val path = "/tmp/dl/spark-tf-connector/test-output.tfrecord"
df.write.format("tfrecords").option("recordType", "Example").mode("overwrite").save(path)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Read exported TFRecords back into a DataFrame
// MAGIC Note that the imported DataFrame matches the original (compare `df.show()` and `importedDf1.show()`)

// COMMAND ----------

//The DataFrame schema is inferred from the TFRecords if no custom schema is provided.
val importedDf1: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").load(path)
importedDf1.show()

//Read TFRecords into DataFrame using custom schema
val importedDf2: DataFrame = spark.read.format("tfrecords").schema(schema).load(path)
importedDf2.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Loading an existing TFRecord dataset into Spark
// MAGIC The example below loads the YouTube-8M dataset into a DataFrame. First, we download the dataset to DBFS:

// COMMAND ----------

// MAGIC %sh
// MAGIC curl -s http://us.data.yt8m.org/1/video_level/train/train-0.tfrecord > /dbfs/tmp/dl/spark-tf-connector/video_level-train-0.tfrecord

// COMMAND ----------

// MAGIC %sh
// MAGIC ls -l /dbfs/tmp/dl/spark-tf-connector/

// COMMAND ----------

// MAGIC %sh
// MAGIC ls -l dbfs:/dbfs/tmp/dl/spark-tf-connector/

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/dl/spark-tf-connector/"))

// COMMAND ----------

// MAGIC %md #### Declare schema and import data into a DataFrame

// COMMAND ----------

//Import Video-level Example dataset into DataFrame
val videoSchema = StructType(List(StructField("video_id", StringType),
                             StructField("labels", ArrayType(IntegerType, true)),
                             StructField("mean_rgb", ArrayType(FloatType, true)),
                             StructField("mean_audio", ArrayType(FloatType, true))))
val videoDf: DataFrame = spark.read.format("tfrecords").schema(videoSchema).option("recordType", "Example")
  .load("dbfs:/tmp/dl/spark-tf-connector/video_level-train-0.tfrecord")
videoDf.show(5)

// COMMAND ----------

// MAGIC %md #### Export data to TFRecords and import it back into a DataFrame
// MAGIC Note that the imported DataFrame (`importedDf1`) matches the original (`videoDf`).

// COMMAND ----------

// Write DataFrame to a tfrecords file
// WARNING: This command will overwrite existing data
videoDf.write.format("tfrecords").option("recordType", "Example").mode("overwrite").save("dbfs:/tmp/dl/spark-tf-connector/youtube-8m-video.tfrecords")

// COMMAND ----------

// Import data back into a DataFrame, verify that it matches the original
val importedDf1: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").schema(videoSchema).load("dbfs:/tmp/dl/spark-tf-connector/youtube-8m-video.tfrecords")
importedDf1.show(5)

// COMMAND ----------

// MAGIC %md #### Remove downloaded data files

// COMMAND ----------

// MAGIC %fs rm -r /tmp/dl/spark-tf-connector/