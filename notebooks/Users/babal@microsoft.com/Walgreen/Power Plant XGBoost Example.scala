// Databricks notebook source
// MAGIC %md # XGBoost with Spark DataFrames
// MAGIC 
// MAGIC ![alt text](http://i.imgur.com/iFZNBVx.png "XGBoost")
// MAGIC #### Before You Start: Build and Install XGBoost
// MAGIC In order to run this notebook, you will need to build and install XGBoost. For instructions on how to do this, please refer to **Installing** and **Testing** sections of the [Databricks XGBoost Docs](https://docs.databricks.com/user-guide/faq/xgboost.html). 

// COMMAND ----------

// MAGIC %md ## Build XGBoost Model & Pipeline
// MAGIC #### Import XGBoost Libraries and Prepare Data

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.{DataUtils, XGBoost}

// COMMAND ----------

sql("drop table if exists power_plant")
case class PowerPlantTable(AT: Double, V : Double, AP : Double, RH : Double, PE : Double)
val powerPlantData = sc.textFile("dbfs:/databricks-datasets/power-plant/data/")
  .map(x => x.split("\t"))
  .filter(line => line(0) != "AT")
  .map(line => PowerPlantTable(line(0).toDouble, line(1).toDouble, line(2).toDouble, line(3).toDouble, line(4).toDouble))
  .toDF
  .write
  .saveAsTable("power_plant")

val dataset = sqlContext.table("power_plant")

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val assembler =  new VectorAssembler()
  .setInputCols(Array("AT", "V", "AP", "RH"))
  .setOutputCol("features")

val vected = assembler.transform(dataset).withColumnRenamed("PE", "label").drop("AT","V","AP","RH")

// COMMAND ----------

val Array(split20, split80) = vected.randomSplit(Array(0.20, 0.80), 1800009193L)
val testSet = split20.cache()
val trainingSet = split80.cache()

// COMMAND ----------

// MAGIC %md #### Train XGBoost Model with Spark DataFrames

// COMMAND ----------

val paramMap = List(
  "eta" -> 0.3,
  "max_depth" -> 6,
  "objective" -> "reg:linear",
  "early_stopping_rounds" ->10).toMap

val xgboostModel = XGBoost.trainWithDataFrame(trainingSet, paramMap, 30, 10, useExternalMemory=true)

// COMMAND ----------

// MAGIC %md #### Evaluate Model
// MAGIC 
// MAGIC You can evaluate the XGBoost model using Evaluators from MLlib.

// COMMAND ----------

val predictions = xgboostModel.transform(testSet)

// COMMAND ----------

display(predictions)

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)

// COMMAND ----------

print ("Root mean squared error: " + rmse)

// COMMAND ----------

// MAGIC %md #### Persist Model

// COMMAND ----------

// MAGIC %sh
// MAGIC ls /tmp/

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/myXgboostModel/"))

// COMMAND ----------

// MAGIC %fs rm -r /tmp/myXgboostModel

// COMMAND ----------

xgboostModel.save("/tmp/myXgboostModel")

// COMMAND ----------

// MAGIC %fs rm -r /mnt/walgreens/myXgboostModel

// COMMAND ----------

xgboostModel.save("/mnt/walgreens/myXgboostModel")

// COMMAND ----------

// MAGIC %fs rm -r /tmp/xgbm

// COMMAND ----------

xgboostModel.booster.saveModel("/tmp/xgbm")

// COMMAND ----------

display(dbutils.fs.ls("file:/tmp/xgbm"))

// COMMAND ----------

// MAGIC %fs cp file:/tmp/xgbm /mnt/walgreens/xgbm

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/walgreens/xgbm"))

// COMMAND ----------

import pickle
//save model
//pickle.dump(xgboostModel, "/tmp/xgbModelsPickle/xgbpickle.model")

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.{Pipeline, PipelineModel}

val xgboostEstimator = new XGBoostEstimator(
        Map[String, Any]("num_round" -> 30, "nworkers" -> 10, "objective" -> "reg:linear", "eta" -> 0.3, "max_depth" -> 6, "early_stopping_rounds" -> 10))

// construct the pipeline       
val pipeline = new Pipeline()
      .setStages(Array(assembler, xgboostEstimator))

// COMMAND ----------

val pipelineData = dataset.withColumnRenamed("PE","label")
val pipelineModel = pipeline.fit(pipelineData)

// COMMAND ----------

display(pipelineModel.transform(pipelineData))

// COMMAND ----------

// MAGIC %md #### Tune Model using MLlib Cross Validation

// COMMAND ----------

import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}

val paramGrid = new ParamGridBuilder()
      .addGrid(xgboostEstimator.maxDepth, Array(4, 7))
      .addGrid(xgboostEstimator.eta, Array(0.1, 0.6))
      .build()

val cv = new CrossValidator()
      .setEstimator(xgboostEstimator)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)

val cvModel = cv.fit(trainingSet)

// COMMAND ----------

cvModel.bestModel.extractParamMap

// COMMAND ----------

// MAGIC %md We can see by running the evaluator below that the model tuning improved RMSE from 3.24 to 2.49.

// COMMAND ----------

val model = cvModel.bestModel.asInstanceOf[ml.dmlc.xgboost4j.scala.spark.XGBoostModel]

// COMMAND ----------

// MAGIC %fs rm -r /tmp/xgbmodel

// COMMAND ----------

// MAGIC %fs rm -r /mnt/walgreens/xgbmodel

// COMMAND ----------

display(dbutils.fs.ls("/mnt/walgreens"))

// COMMAND ----------

model.save("/tmp/xgbmodel")

// COMMAND ----------

//display(dbutils.fs.ls("/mnt/walgreens"))
model.save("/mnt/walgreens/xgbmodel")

// COMMAND ----------

val results = cvModel.transform(testSet)
evaluator.evaluate(results)

// COMMAND ----------

display(results)

// COMMAND ----------

// MAGIC %md #### Persist Tuned Model and Pipeline

// COMMAND ----------

// MAGIC %fs rm -r /tmp/xgboostTunedModel

// COMMAND ----------

// MAGIC %fs rm -r /mnt/walgreens/xgboostTunedModel

// COMMAND ----------

// Save Best Model from Cross Validation
cvModel.bestModel.asInstanceOf[ml.dmlc.xgboost4j.scala.spark.XGBoostModel].save("/tmp/xgboostTunedModel")

// COMMAND ----------

// Save Best Model from Cross Validation
cvModel.bestModel.asInstanceOf[ml.dmlc.xgboost4j.scala.spark.XGBoostModel].save("/mnt/walgreens/xgboostTunedModel")

// COMMAND ----------

// Or you could save the entire ML Pipeine
pipelineModel.write.overwrite().save("/tmp/xgPipeline")

// COMMAND ----------

// Or you could save the entire ML Pipeine
pipelineModel.write.overwrite().save("/mnt/walgreens/xgPipeline")

// COMMAND ----------

import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.mleap.core.feature.{OneHotEncoderModel, StringIndexerModel}
import ml.combust.mleap.core.regression.LinearRegressionModel
//import ml.combust.mleap.runtime.transformer.Pipeline
import ml.combust.mleap.runtime.transformer.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import ml.combust.mleap.runtime.transformer.regression.LinearRegression
import org.apache.spark.ml.linalg.Vectors
import ml.combust.mleap.runtime.MleapSupport._
import resource._

// COMMAND ----------

import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.mleap.SparkUtil
import ml.combust.mleap.spark.SparkSupport._
import resource._

// COMMAND ----------

//import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
//import org.apache.spark.ml.{Pipeline, PipelineModel}

// COMMAND ----------

pipelineModel.stages

// COMMAND ----------

// MAGIC %fs rm /tmp/simple-json1.zip

// COMMAND ----------

for(bundle <- managed(BundleFile("jar:file:/tmp/simple-json1.zip"))) {
  pipelineModel.writeBundle.format(SerializationFormat.Json).save(bundle)
}

// COMMAND ----------

dbutils.fs.cp("file:/tmp/simple-json1.zip", "dbfs:/mnt/walgreens/model/simple-json1.zip")

// COMMAND ----------

display(dbutils.fs.ls("/tmp/xgPipeline"))

// COMMAND ----------

display(dbutils.fs.ls("/mnt/walgreens/xgPipeline"))

// COMMAND ----------

// MAGIC %md #### Load XGBoost Pipeline

// COMMAND ----------

import org.apache.spark.ml.PipelineModel

//val loadedPipeline = PipelineModel.load("/tmp/xgPipeline")
val loadedPipeline = PipelineModel.load("/mnt/walgreens/xgPipeline")

// COMMAND ----------

loadedPipeline

// COMMAND ----------

// MAGIC %fs ls /tmp/xgPipeline/metadata

// COMMAND ----------

// MAGIC %fs ls /tmp/xgPipeline/stages

// COMMAND ----------

// MAGIC %sh 
// MAGIC rm -rf /tmp/mleap_scala_model_export/
// MAGIC mkdir /tmp/mleap_scala_model_export/

// COMMAND ----------

// MAGIC %sh
// MAGIC ls /tmp/mleap_scala_model_export/

// COMMAND ----------

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml.bundle.SparkBundleContext
import resource._

// COMMAND ----------

// MAGIC %sh ls /tmp/mleap_scala_model_export/

// COMMAND ----------


implicit val context = SparkBundleContext().withDataset(results)
//save our pipeline to a zip file
//MLeap can save a file to any supported java.nio.FileSystem
(for(modelFile <- managed(BundleFile("jar:file:/tmp/mleap_scala_model_export/20news_pipeline-json.zip"))) yield {
  model.writeBundle.save(modelFile)(context)
}).tried.get

// COMMAND ----------

// MAGIC %sh ls /tmp/mleap_scala_model_export/

// COMMAND ----------

//dbutils.fs.cp("file:/tmp/mleap_scala_model_export/20news_pipeline-json.zip", "dbfs:/data/20news_pipeline-json.zip")
display(dbutils.fs.ls("dbfs:/data"))

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

for(bundle <- managed(BundleFile("jar:file:/tmp/simple-json.zip"))) {
  loadedPipeline.writeBundle.format(SerializationFormat.Json).save(bundle)
}

// COMMAND ----------



// COMMAND ----------

display(dbutils.fs.ls("file:/tmp/simple-json.zip"))

// COMMAND ----------

dbutils.fs.cp("file:/tmp/simple-json.zip", "dbfs:/mnt/walgreens/model/simple-json.zip")