# Databricks notebook source
dbutils.fs.ls("file:/tmp/")

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/myXgboostModel/"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/xgboostTunedModel/"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/xgPipeline/"))

# COMMAND ----------

import org.apache.spark.ml.PipelineModel

loadedPipeline = PipelineModel.load("/tmp/xgPipeline")