// Databricks notebook source
display(dbutils.fs.ls("dbfs:/tmp/xgPipeline/"))

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

display(testSet)

// COMMAND ----------

val pipelineData = dataset.withColumnRenamed("PE","label")

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/xgPipeline/"))

// COMMAND ----------

import org.apache.spark.ml.PipelineModel

val loadedPipeline = PipelineModel.load("/tmp/xgPipeline")

// COMMAND ----------

loadedPipeline

// COMMAND ----------

val predictions = loadedPipeline.transform(pipelineData)

// COMMAND ----------

display(predictions)

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/xgbmodel"))

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.XGBoost
import ml.dmlc.xgboost4j.scala.Booster

val model = XGBoost.loadModel("dbfs:/tmp/xgbmodel.model")
//val xGBoost = new XGBoostRegressionModel(model)
//Booster booster = XGBoost.loadModel("/tmp/xgbmodel.model")

// COMMAND ----------

