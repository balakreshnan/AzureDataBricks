// Databricks notebook source
import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.mleap.SparkUtil
import ml.combust.mleap.spark.SparkSupport._
import resource._

// COMMAND ----------

display(dbutils.fs.ls("file:/tmp/simple-json.zip"))

// COMMAND ----------

// Deserialize a zip bundle
// Use Scala ARM to make sure resources are managed properly
val zipBundle = (for(bundle <- managed(BundleFile("jar:file:/tmp/simple-json.zip"))) yield {
  bundle.loadSparkBundle().get
}).opt.get

// COMMAND ----------

display(zipBundle)