# Databricks notebook source
# MAGIC %md
# MAGIC # Predict Weather Related Flight Delays

# COMMAND ----------

# Subset of Flight & Weather Data
air_file_loc = "wasb://data@cdspsparksamples.blob.core.windows.net/Airline/AirlineSubsetCsv"
weather_file_loc = "wasb://data@cdspsparksamples.blob.core.windows.net/Airline/WeatherSubsetCsv"

# COMMAND ----------

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import numpy as np
import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Air and Weather Data

# COMMAND ----------

## Read Airline On-time data
air = spark.read.csv(path=air_file_loc, header=True, inferSchema=True)
air.createOrReplaceTempView("airline")

## Read Weather data
weather = spark.read.csv(path=weather_file_loc, header=True, inferSchema=True)
weather.createOrReplaceTempView("weather")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean and Merge Data

# COMMAND ----------

# COUNT FLIGHTS BY AIRPORT
spark.sql("SELECT ORIGIN, COUNT(*) as CTORIGIN FROM airline GROUP BY ORIGIN").createOrReplaceTempView("countOrigin")
spark.sql("SELECT DEST, COUNT(*) as CTDEST FROM airline GROUP BY DEST").createOrReplaceTempView("countDest")

# COMMAND ----------

## FILTER FOR AIRPORTS WHICH HAVE VERY FEW FLIGHTS (<100)
sqlStatement = """SELECT ARR_DEL15 as ArrDel15,  YEAR  as Year,
                  MONTH as Month, DAY_OF_MONTH as DayOfMonth, DAY_OF_WEEK as DayOfWeek,
                  UNIQUE_CARRIER as Carrier, ORIGIN_AIRPORT_ID as OriginAirportID, ORIGIN,
                  DEST_AIRPORT_ID as DestAirportID, DEST, floor(CRS_DEP_TIME/100) as CRSDepTime,
                  floor(CRS_ARR_TIME/100) as CRSArrTime
                  FROM airline
                  WHERE ARR_DEL15 in ('0.0', '1.0')
                  AND ORIGIN IN (SELECT DISTINCT ORIGIN FROM countOrigin where CTORIGIN > 100)
                  AND DEST IN (SELECT DISTINCT DEST FROM countDest where CTDEST > 100) """
airCleaned = spark.sql(sqlStatement)

airCleaned.createOrReplaceTempView("airCleaned")

# COMMAND ----------

## CLEAN WEATHER DATA WITH QUERY
sqlStatement = """SELECT AdjustedYear, AdjustedMonth, AdjustedDay, AdjustedHour, AirportID,
                            avg(Visibility) as Visibility, avg(DryBulbCelsius) as DryBulbCelsius, avg(DewPointCelsius) as DewPointCelsius,
                            avg(RelativeHumidity) as RelativeHumidity, avg(WindSpeed) as WindSpeed, avg(Altimeter) as Altimeter
                            FROM weather
                            GROUP BY AdjustedYear, AdjustedMonth, AdjustedDay, AdjustedHour, AirportID"""
weatherCleaned = spark.sql(sqlStatement)

weatherCleaned.createOrReplaceTempView("weatherCleaned")

# COMMAND ----------

# JOIN and FILTER
sqlStatement = """SELECT a.ArrDel15, a.Year, a.Month, a.DayOfMonth, a.DayOfWeek, a.Carrier, a.OriginAirportID, \
                              a.ORIGIN, a.DestAirportID, a.DEST, a.CRSDepTime, b.Visibility as VisibilityOrigin, \
                              b.DryBulbCelsius as DryBulbCelsiusOrigin, b.DewPointCelsius as DewPointCelsiusOrigin,
                              b.RelativeHumidity as RelativeHumidityOrigin, b.WindSpeed as WindSpeedOrigin, \
                              b.Altimeter as AltimeterOrigin, c.Visibility as VisibilityDest, \
                              c.DryBulbCelsius as DryBulbCelsiusDest, c.DewPointCelsius as DewPointCelsiusDest,
                              c.RelativeHumidity as RelativeHumidityDest, c.WindSpeed as WindSpeedDest, \
                              c.Altimeter as AltimeterDest
                              FROM airCleaned a, weatherCleaned b, weatherCleaned c
                              WHERE a.Year = b.AdjustedYear and a.Year = c.AdjustedYear
                              and a.Month = b.AdjustedMonth and a.Month = c.AdjustedMonth
                              and a.DayofMonth = b.AdjustedDay and a.DayofMonth = c.AdjustedDay
                              and a.CRSDepTime= b.AdjustedHour and a.CRSDepTime = c.AdjustedHour
                              and a.OriginAirportID = b.AirportID and a.DestAirportID = c.AirportID"""

joined = spark.sql(sqlStatement).filter("VisibilityOrigin is not NULL and DryBulbCelsiusOrigin is not NULL \
                and DewPointCelsiusOrigin is not NULL and RelativeHumidityOrigin is not NULL \
                and WindSpeedOrigin is not NULL and AltimeterOrigin is not NULL \
                and VisibilityDest is not NULL and DryBulbCelsiusDest is not NULL \
                and DewPointCelsiusDest is not NULL and RelativeHumidityDest is not NULL \
                and WindSpeedDest is not NULL and AltimeterDest is not NULL \
                and ORIGIN is not NULL and DEST is not NULL \
                and OriginAirportID is not NULL and DestAirportID is not NULL \
                and CRSDepTime is not NULL and Year is not NULL and Month is not NULL \
                and DayOfMonth is not NULL and DayOfWeek is not NULL and Carrier is not NULL")

joined.createOrReplaceTempView("joined")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from joined

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Train and Validation Sets

# COMMAND ----------

# Training Data
train_filename = "/mnt/training-adl/data-science-demo/data/training"
train = spark.sql("SELECT * from joined WHERE Year = 2011")
train.write.mode("overwrite").parquet(train_filename)

# Validation Data
validation_filename = "/mnt/training-adl/data-science-demo/data/validation"
validation = spark.sql("SELECT * from joined WHERE Year = 2012")
validation.write.mode("overwrite").parquet(validation_filename)

# COMMAND ----------

#Ingest Training Data
train_df = spark.read.parquet(train_filename)
train_df.cache()
train_df.createOrReplaceTempView("train")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering & Data Prep

# COMMAND ----------

trainingFraction = 0.75; testingFraction = (1-trainingFraction);
seed = 1234;

# SPLIT SAMPLED DATA-FRAME INTO TRAIN/TEST, WITH A RANDOM COLUMN ADDED FOR DOING CV (SHOWN LATER)
trainPartition, testPartition = train_df.randomSplit([trainingFraction, testingFraction], seed=seed);

trainPartition.createOrReplaceTempView("TrainPartition")
testPartition.createOrReplaceTempView("TestPartition")

# COMMAND ----------

trainPartitionFilt = trainPartition.filter("ArrDel15 is not NULL and DayOfMonth is not NULL and DayOfWeek is not NULL \
                and Carrier is not NULL and OriginAirportID is not NULL and DestAirportID is not NULL \
                and CRSDepTime is not NULL and VisibilityOrigin is not NULL and DryBulbCelsiusOrigin is not NULL \
                and DewPointCelsiusOrigin is not NULL and RelativeHumidityOrigin is not NULL \
                and WindSpeedOrigin is not NULL and AltimeterOrigin is not NULL \
                and VisibilityDest is not NULL and DryBulbCelsiusDest is not NULL \
                and DewPointCelsiusDest is not NULL and RelativeHumidityDest is not NULL \
                and WindSpeedDest is not NULL and AltimeterDest is not NULL ")

trainPartitionFilt.createOrReplaceTempView("TrainPartitionFilt")

testPartitionFilt = testPartition.filter("ArrDel15 is not NULL and DayOfMonth is not NULL and DayOfWeek is not NULL \
                and Carrier is not NULL and OriginAirportID is not NULL and DestAirportID is not NULL \
                and CRSDepTime is not NULL and VisibilityOrigin is not NULL and DryBulbCelsiusOrigin is not NULL \
                and DewPointCelsiusOrigin is not NULL and RelativeHumidityOrigin is not NULL \
                and WindSpeedOrigin is not NULL and AltimeterOrigin is not NULL \
                and VisibilityDest is not NULL and DryBulbCelsiusDest is not NULL \
                and DewPointCelsiusDest is not NULL and RelativeHumidityDest is not NULL \
                and WindSpeedDest is not NULL and AltimeterDest is not NULL") \
                .filter("OriginAirportID IN (SELECT distinct OriginAirportID FROM TrainPartitionFilt) \
                    AND ORIGIN IN (SELECT distinct ORIGIN FROM TrainPartitionFilt) \
                    AND DestAirportID IN (SELECT distinct DestAirportID FROM TrainPartitionFilt) \
                    AND DEST IN (SELECT distinct DEST FROM TrainPartitionFilt) \
                    AND Carrier IN (SELECT distinct Carrier FROM TrainPartitionFilt) \
                    AND CRSDepTime IN (SELECT distinct CRSDepTime FROM TrainPartitionFilt) \
                    AND DayOfMonth in (SELECT distinct DayOfMonth FROM TrainPartitionFilt) \
                    AND DayOfWeek in (SELECT distinct DayOfWeek FROM TrainPartitionFilt)")

testPartitionFilt.createOrReplaceTempView("TestPartitionFilt")

# COMMAND ----------

# TRANSFORM FEATURES
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer, Bucketizer, Binarizer

sI0 = StringIndexer(inputCol = 'ArrDel15', outputCol = 'ArrDel15_ind'); bin0 = Binarizer(inputCol = 'ArrDel15_ind', outputCol = 'ArrDel15_bin', threshold = 0.5);
sI1 = StringIndexer(inputCol="Carrier", outputCol="Carrier_ind");
transformPipeline = Pipeline(stages=[sI0, bin0, sI1]);

transformedTrain = transformPipeline.fit(trainPartition).transform(trainPartitionFilt)
transformedTest = transformPipeline.fit(trainPartition).transform(testPartitionFilt)

transformedTrain.cache()
transformedTest.cache()

# COMMAND ----------

from pyspark.ml.feature import RFormula

## DEFINE REGRESSION FURMULA
regFormula = RFormula(formula="ArrDel15_ind ~ \
                        DayOfMonth + DayOfWeek + Carrier_ind + OriginAirportID + DestAirportID + CRSDepTime \
                        + VisibilityOrigin + DryBulbCelsiusOrigin + DewPointCelsiusOrigin \
                        + RelativeHumidityOrigin + WindSpeedOrigin + AltimeterOrigin \
                        + VisibilityDest + DryBulbCelsiusDest + DewPointCelsiusDest \
                        + RelativeHumidityDest + WindSpeedDest + AltimeterDest");

## DEFINE INDEXER FOR CATEGORIAL VARIABLES
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=250)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from sklearn.metrics import roc_curve,auc

## DEFINE ELASTIC NET REGRESSOR
eNet = LogisticRegression(featuresCol="indexedFeatures", maxIter=25, regParam=0.01, elasticNetParam=0.5)

## TRAINING PIPELINE: Fit model, with formula and other transformations
model = Pipeline(stages=[regFormula, featureIndexer, eNet]).fit(transformedTrain)

# SAVE MODEL
datestamp = datetime.datetime.now().strftime('%m-%d-%Y-%s');
fileName = "logisticRegModel_" + datestamp;
model_filename = "/mnt/training-adl/data-science-demo/" + fileName;
model.save(model_filename)

## Evaluate model on test set
predictions = model.transform(transformedTest)
predictionAndLabels = predictions.select("label","prediction").rdd
predictions.select("label","probability").createOrReplaceTempView("tmp_results")

metrics = BinaryClassificationMetrics(predictionAndLabels)
print("Area under ROC = %s" % metrics.areaUnderROC)

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor

## DEFINE GRADIENT BOOSTING TREE CLASSIFIER
gBT = GBTRegressor(featuresCol="indexedFeatures", maxIter=10, maxBins = 250)

## TRAINING PIPELINE: Fit model, with formula and other transformations
model = Pipeline(stages=[regFormula, featureIndexer, gBT]).fit(transformedTrain)

# SAVE MODEL
datestamp = datetime.datetime.now().strftime('%m-%d-%Y-%s');
fileName = "gbtModel_" + datestamp;
model_filename = "/mnt/training-adl/data-science-demo/" + fileName;
model.save(model_filename)

## Evaluate model on test set
predictions = model.transform(transformedTest)
predictionAndLabels = predictions.select("label","prediction").rdd

metrics = BinaryClassificationMetrics(predictionAndLabels)
print("Area under ROC = %s" % metrics.areaUnderROC)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

## DEFINE RANDOM FOREST MODELS
## DEFINE RANDOM FOREST CLASSIFIER
randForest = RandomForestClassifier(featuresCol = 'indexedFeatures', labelCol = 'label', numTrees=20, \
                                   maxDepth=6, maxBins=250)


## DEFINE MODELING PIPELINE, INCLUDING FORMULA, FEATURE TRANSFORMATIONS, AND ESTIMATOR
pipeline = Pipeline(stages=[regFormula, featureIndexer, randForest])

## DEFINE PARAMETER GRID FOR RANDOM FOREST
paramGrid = ParamGridBuilder() \
    .addGrid(randForest.numTrees, [10, 25, 50]) \
    .addGrid(randForest.maxDepth, [3, 5, 7]) \
    .build()

## DEFINE CROSS VALIDATION
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(metricName="areaUnderROC"),
                          numFolds=3)

## TRAIN MODEL USING CV
cvModel = crossval.fit(transformedTrain)

## Evaluate model on test set
predictions = cvModel.transform(transformedTest)
predictionAndLabels = predictions.select("label","prediction").rdd
metrics = BinaryClassificationMetrics(predictionAndLabels)
print("Area under ROC = %s" % metrics.areaUnderROC)

## SAVE THE BEST MODEL
datestamp = datetime.datetime.now().strftime('%m-%d-%Y-%s');
fileName = "CV_RandomForestRegressionModel_" + datestamp;
model_filename = "/mnt/training-adl/data-science-demo/" + fileName;
cvModel.bestModel.save(model_filename)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Operationalize

# COMMAND ----------

from pyspark.ml import PipelineModel

savedModel = PipelineModel.load(model_filename)

## Evaluate model on test set
predictions = savedModel.transform(transformedTest)
predictionAndLabels = predictions.select("label","prediction").rdd
metrics = BinaryClassificationMetrics(predictionAndLabels)
print("Area under ROC = %s" % metrics.areaUnderROC)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Model

# COMMAND ----------

## READ IN DATA FRAME FROM PARQUET
validPartition = spark.read.parquet(validation_filename)

validPartitionFilt = validPartition.filter("ArrDel15 is not NULL and DayOfMonth is not NULL and DayOfWeek is not NULL \
                and Carrier is not NULL and OriginAirportID is not NULL and DestAirportID is not NULL \
                and CRSDepTime is not NULL and VisibilityOrigin is not NULL and DryBulbCelsiusOrigin is not NULL \
                and DewPointCelsiusOrigin is not NULL and RelativeHumidityOrigin is not NULL \
                and WindSpeedOrigin is not NULL and AltimeterOrigin is not NULL \
                and VisibilityDest is not NULL and DryBulbCelsiusDest is not NULL \
                and DewPointCelsiusDest is not NULL and RelativeHumidityDest is not NULL \
                and WindSpeedDest is not NULL and AltimeterDest is not NULL") \
                .filter("OriginAirportID IN (SELECT distinct OriginAirportID FROM TrainPartitionFilt) \
                    AND ORIGIN IN (SELECT distinct ORIGIN FROM TrainPartitionFilt) \
                    AND DestAirportID IN (SELECT distinct DestAirportID FROM TrainPartitionFilt) \
                    AND DEST IN (SELECT distinct DEST FROM TrainPartitionFilt) \
                    AND Carrier IN (SELECT distinct Carrier FROM TrainPartitionFilt) \
                    AND CRSDepTime IN (SELECT distinct CRSDepTime FROM TrainPartitionFilt) \
                    AND DayOfMonth in (SELECT distinct DayOfMonth FROM TrainPartitionFilt) \
                    AND DayOfWeek in (SELECT distinct DayOfWeek FROM TrainPartitionFilt)")

# COMMAND ----------

transformedValid = transformPipeline.fit(trainPartition).transform(validPartitionFilt)

# COMMAND ----------

savedModel = PipelineModel.load(model_filename)
predictions = savedModel.transform(transformedValid)
predictionAndLabels = predictions.select("label","prediction").rdd
metrics = BinaryClassificationMetrics(predictionAndLabels)
print("Area under ROC = %s" % metrics.areaUnderROC)