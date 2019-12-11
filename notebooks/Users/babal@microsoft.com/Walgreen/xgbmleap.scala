// Databricks notebook source
import org.apache.spark.ml.PipelineModel

//val loadedPipeline = PipelineModel.load("/tmp/xgPipeline")
val loadedPipeline = PipelineModel.load("/mnt/walgreens/xgPipeline")

// COMMAND ----------

import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.mleap.core.feature.{OneHotEncoderModel, StringIndexerModel}
import ml.combust.mleap.core.regression.LinearRegressionModel
import ml.combust.mleap.runtime.transformer.Pipeline
import ml.combust.mleap.runtime.transformer.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import ml.combust.mleap.runtime.transformer.regression.LinearRegression
import org.apache.spark.ml.linalg.Vectors
import ml.combust.mleap.runtime.MleapSupport._
import resource._

// COMMAND ----------

import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import ml.combust.mleap.core.regression.LinearRegressionModel
import ml.combust.mleap.runtime.transformer.regression.LinearRegression
import org.apache.spark.ml.mleap.SparkUtil
import ml.combust.mleap.spark.SparkSupport._
import resource._

// COMMAND ----------

for(bundle <- managed(BundleFile("jar:file:/tmp/simple-json.zip"))) {
  loadedPipeline.writeBundle.format(SerializationFormat.Json).save(bundle)
}

// COMMAND ----------

display(dbutils.fs.ls("file:/tmp/simple-json.zip"))

// COMMAND ----------

dbutils.fs.cp("file:/tmp/simple-json.zip", "dbfs:/mnt/walgreens/model/simple-json.zip")