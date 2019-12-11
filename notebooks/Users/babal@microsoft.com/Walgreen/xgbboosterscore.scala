// Databricks notebook source
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
import ml.dmlc.xgboost4j.scala.XGBoost
import ml.dmlc.xgboost4j.scala.DMatrix

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/xgbm"))

// COMMAND ----------

val xgboost = XGBoost.loadModel("file:/tmp/xgbm")

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel

val xgbClassificationModelPath = "/tmp/xgbmodel"

//val xgbClassificationModel2 = XGBoostRegressionModel.load(xgbClassificationModelPath)
//xgbClassificationModel2.transform(xgbInput)

// COMMAND ----------

