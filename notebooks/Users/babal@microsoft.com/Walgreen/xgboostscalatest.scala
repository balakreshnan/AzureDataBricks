// Databricks notebook source
display(dbutils.fs.ls("dbfs:/tmp/myXgboostModel/"))

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/xgboostTunedModel/"))

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/xgPipeline/"))

// COMMAND ----------

dbutils.fs.ls("/mnt/walgreens")

// COMMAND ----------

//dbutils.fs.unmount("/mnt/walgreens")

// COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://walgreens@dewsa.blob.core.windows.net/",
  mountPoint = "/mnt/walgreens",
  extraConfigs = Map("fs.azure.account.key.dewsa.blob.core.windows.net" -> "TDjjpKGq75D4t0fy4D5fNTJrnzlrtCKxB9MIipsfQvLWsR3mScAswOZTDOMi0JSXi1fb2ga5ly4rUAQ+WUvETQ=="))

// COMMAND ----------

display(dbutils.fs.ls("/mnt/walgreens"))

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

display(dbutils.fs.ls("dbfs:/mnt/walgreens/xgPipeline/"))

// COMMAND ----------

// MAGIC %fs cp -r dbfs:/tmp/xgPipeline/ dbfs:/mnt/walgreens/

// COMMAND ----------

dbutils.fs.cp("dbfs:/tmp/xgPipeline/", "/mnt/walgreens/")

// COMMAND ----------

import org.apache.spark.ml.PipelineModel

//val loadedPipeline = PipelineModel.load("/tmp/xgPipeline")
val loadedPipeline = PipelineModel.load("/mnt/walgreens/xgPipeline")


// COMMAND ----------

loadedPipeline

// COMMAND ----------

val predictions = loadedPipeline.transform(pipelineData)

// COMMAND ----------

display(predictions)

// COMMAND ----------

predictions.select("label", "prediction", "AT","V", "AP", "RH").show()

// COMMAND ----------

val resultRDD = predictions.select("prediction", "indexedLabel").rdd.map { case Row(prediction: Double, label: Double) => (prediction, label) }
    val (precision, recall, f1) = MultiClassEvaluation.multiClassEvaluate(resultRDD)
    println("\n\n========= ???? ==========")
    println(s"\n??????$precision")
    println(s"??????$recall")
    println(s"F1??$f1")

// COMMAND ----------

// MAGIC %fs ls /tmp/xgPipeline/stages

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/data"))

// COMMAND ----------

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import resource._

val zipBundle = (for(bundle <- managed(BundleFile("jar:file:/tmp/mleap_scala_model_export/20news_pipeline-json.zip"))) yield {
  bundle.loadSparkBundle().get
}).opt.get

// COMMAND ----------

val loadedModel = zipBundle.root

// COMMAND ----------

val test_df = spark.read.parquet("/databricks-datasets/news20.binary/data-001/test")
  .select("text", "topic")
test_df.cache()
display(test_df)

// COMMAND ----------

val exampleResults = loadedModel.transform(test_df)

display(exampleResults)

// COMMAND ----------

