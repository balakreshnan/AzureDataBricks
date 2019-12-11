// Databricks notebook source
display(dbutils.fs.ls("/mnt/walgreens"))

// COMMAND ----------

//load the data set to process which is the data to score against the model
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

display(vected)

// COMMAND ----------

val pipelineData = dataset.withColumnRenamed("PE","label")

// COMMAND ----------

import org.apache.spark.ml.PipelineModel

val loadedPipeline = PipelineModel.load("/mnt/walgreens/xgPipeline")

// COMMAND ----------

display(pipelineData)

// COMMAND ----------

val predictions = loadedPipeline.transform(pipelineData)

// COMMAND ----------

display(predictions)

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.{DataUtils, XGBoost}

// COMMAND ----------

val xgb = XGBoost.load("dbfs:/mnt/walgreens/xgbm")

// COMMAND ----------

// MAGIC %python
// MAGIC import xgboost as xgb
// MAGIC bst = xgb.Booster({'nthread': 4})
// MAGIC bst.load_model("dbfs:/mnt/walgreens/xgbm")

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel

val xgbregressionmodel = XGBoostRegressionModel.load("/tmp/xgbm")
xgbregressionmodel.transform(pipelineData)