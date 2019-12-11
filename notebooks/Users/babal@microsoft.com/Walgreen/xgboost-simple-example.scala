// Databricks notebook source
import ml.dmlc.xgboost4j.scala.spark.{XGBoost}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.DenseVector

// COMMAND ----------

val trainRDD = sc.parallelize(Seq(
  LabeledPoint(1.0, new DenseVector(Array(2.0, 3.0, 4.0))), 
  LabeledPoint(0.0, new DenseVector(Array(5.0, 5.0, 5.0))), 
  LabeledPoint(1.0, new DenseVector(Array(2.0, 3.0, 4.0))), 
  LabeledPoint(0.0, new DenseVector(Array(5.0, 5.0, 5.0))), 
  LabeledPoint(1.0, new DenseVector(Array(2.0, 3.0, 4.0))), 
  LabeledPoint(0.0, new DenseVector(Array(5.0, 5.0, 5.0))), 
  LabeledPoint(1.0, new DenseVector(Array(2.0, 3.0, 4.0))), 
  LabeledPoint(0.0, new DenseVector(Array(5.0, 5.0, 5.0))), 
  LabeledPoint(1.0, new DenseVector(Array(2.0, 3.0, 4.0))), 
  LabeledPoint(0.0, new DenseVector(Array(5.0, 5.0, 5.0))), 
  LabeledPoint(1.0, new DenseVector(Array(2.0, 3.0, 4.0))), 
  LabeledPoint(1.0, new DenseVector(Array(2.0, 3.0, 4.0))), 
  LabeledPoint(0.0, new DenseVector(Array(5.0, 5.0, 5.0)))
), 4)

// COMMAND ----------

val trainRDDofRows = trainRDD.map{ labeledPoint => Row(labeledPoint.label, labeledPoint.features)}

// COMMAND ----------

import org.apache.spark.sql.types.{StructType, DoubleType, StructField}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
val trainSchema = StructType(Array(
                    StructField("label", DoubleType),
                    StructField("features", VectorType)))
val trainDF = spark.createDataFrame(trainRDDofRows, trainSchema)

// COMMAND ----------

display(trainDF)

// COMMAND ----------

val paramMap = List(
  "eta" -> 0.1f,
  "max_depth" -> 2,
  "objective" -> "binary:logistic").toMap

// COMMAND ----------

val xgboostModelRDD = XGBoost.trainWithRDD(trainRDD, paramMap, 1, 4, useExternalMemory=true)

// COMMAND ----------

val xgboostModelDF = XGBoost.trainWithDataFrame(trainDF, paramMap, 1, 4, useExternalMemory = true)

// COMMAND ----------

val xgboostPredictionRDD = xgboostModelRDD.predict(trainRDD.map{x => x.features})

// COMMAND ----------

xgboostPredictionRDD.collect()

// COMMAND ----------

val xgboostPredictionDF = xgboostModelDF.transform(trainDF)

// COMMAND ----------

xgboostPredictionDF.collect()