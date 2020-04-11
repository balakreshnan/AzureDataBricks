// Databricks notebook source
import ml.dmlc.xgboost4j.scala.spark.{DataUtils, XGBoost}

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/walgreens/xgbm"))