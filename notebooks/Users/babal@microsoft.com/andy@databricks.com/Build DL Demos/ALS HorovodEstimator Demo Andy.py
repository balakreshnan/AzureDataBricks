# Databricks notebook source
# MAGIC %md-sandbox 
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs110x/movie-camera.png" style="float:right; height: 200px; margin: 10px; border: 1px solid #ddd; border-radius: 15px 15px 15px 15px; padding: 10px"/>
# MAGIC 
# MAGIC # Scalable Movie Recommendations
# MAGIC 
# MAGIC There are a few options to train neural networks in a distributed manner:
# MAGIC * Dist-keras (CERN)
# MAGIC * TensorFlowOnSpark (Yahoo)
# MAGIC * BigDL (Intel)
# MAGIC * Horovod (Uber)
# MAGIC 
# MAGIC Databricks recommends using __Horovod__ for distributed neural network training (it comes installed with the ML Runtime).
# MAGIC 
# MAGIC Here, we will use 1 million movie ratings from the [MovieLens stable benchmark rating dataset](http://grouplens.org/datasets/movielens/)

# COMMAND ----------

moviesDF = spark.read.parquet("dbfs:/mnt/training/movielens/movies.parquet/")
ratingsDF = spark.read.parquet("dbfs:/mnt/training/movielens/ratings.parquet/")

ratingsDF.cache()
moviesDF.cache()

ratingsCount = ratingsDF.count()
moviesCount = moviesDF.count()

print('There are %s ratings and %s movies in the datasets' % (ratingsCount, moviesCount))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a quick look at some of the data in the two DataFrames.

# COMMAND ----------

display(moviesDF)

# COMMAND ----------

display(ratingsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Part 2: Collaborative Filtering**
# MAGIC 
# MAGIC Let's start by splitting our data into a training and test set.

# COMMAND ----------

seed=42
(trainingDF, testDF) = ratingsDF.randomSplit([0.8, 0.2], seed=seed)

print('Training: {0}, test: {1}'.format(trainingDF.count(), testDF.count()))

# COMMAND ----------

# MAGIC %md ## ALS
# MAGIC 
# MAGIC ![factorization](http://spark-mooc.github.io/web-assets/images/matrix_factorization.png)

# COMMAND ----------

from pyspark.ml.recommendation import ALS

als = (ALS()
       .setUserCol("userId")
       .setItemCol("movieId")
       .setRatingCol("rating")
       .setPredictionCol("prediction")
       .setMaxIter(2)
       .setSeed(seed)
       .setRegParam(0.1)
       .setColdStartStrategy("drop")
       .setRank(12))

alsModel = als.fit(trainingDF)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regEval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="mse")

predictedTestDF = alsModel.transform(testDF)

testMse = regEval.evaluate(predictedTestDF)

print('The model had a MSE on the test set of {0}'.format(testMse))

# COMMAND ----------

# MAGIC %md ## Deep Learning

# COMMAND ----------

userFactors = alsModel.userFactors.selectExpr("id as userId", "features as uFeatures")
itemFactors = alsModel.itemFactors.selectExpr("id as movieId", "features as iFeatures")
joinedTrainDF = trainingDF.join(itemFactors, on="movieId").join(userFactors, on="userId")
joinedTestDF = testDF.join(itemFactors, on="movieId").join(userFactors, on="userId")

# COMMAND ----------

display(joinedTrainDF)

# COMMAND ----------

from itertools import chain
from pyspark.sql.functions import *
from pyspark.sql.types import *

def concat_arrays(*args):
    return list(chain(*args))
    
concat_arrays_udf = udf(concat_arrays, ArrayType(FloatType()))

concatTrainDF = (joinedTrainDF
                 .select('userId', 'movieId', concat_arrays_udf(col("iFeatures"), col("uFeatures")).alias("features"),
                         col('rating').cast("float")))
concatTestDF = (joinedTestDF
                .select('userId', 'movieId', concat_arrays_udf(col("iFeatures"), col("uFeatures")).alias("features"), 
                        col('rating').cast("float")))

# COMMAND ----------

display(concatTrainDF.limit(10))

# COMMAND ----------

# MAGIC %md Define the DL model to train in a Python function.
# MAGIC 
# MAGIC Our tf.estimator-style `model_fn` ([see TensorFlow docs](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)) works by:
# MAGIC 1. Defining the model's network structure, then
# MAGIC 2. Specifying the model's output on a single batch of data during training, eval, and prediction (inference) phases.
# MAGIC 
# MAGIC **Note**: If you have a single-machine `model_fn`, you can prepare it for distributed training with a one-line code change. Simply wrap your optimizer in a `HorovodDistributedOptimizer`, as in the example below.

# COMMAND ----------

import tensorflow as tf
import horovod.tensorflow as hvd

tf.set_random_seed(seed=40)

def model_fn(features, labels, mode, params, config):
    feat_cols = [tf.feature_column.numeric_column(key="features", shape=(24,))]
    regressor = tf.estimator.DNNRegressor(
      hidden_units=[params["hidden_layer1"], params["hidden_layer2"]],
      feature_columns=feat_cols,
      optimizer=hvd.DistributedOptimizer(tf.train.AdamOptimizer(params["learning_rate"] * hvd.size())))
    estimator_spec = regressor.model_fn(features, labels, mode, config)
    export_outputs = estimator_spec.export_outputs
    if export_outputs is not None:
      export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs["predict"]
    return tf.estimator.EstimatorSpec(mode=mode, loss=estimator_spec.loss, train_op=estimator_spec.train_op,
                                      export_outputs=export_outputs, training_hooks=estimator_spec.training_hooks, predictions=estimator_spec.predictions)

# COMMAND ----------

import time

trainValDF = concatTrainDF.withColumn("isVal", when(rand() > 0.8, True).otherwise(False))

model_dir = "/tmp/horovodDemo/" + str(int(time.time()))
print(model_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch model training

# COMMAND ----------

from sparkdl.estimators.horovod_estimator.estimator import HorovodEstimator

est = HorovodEstimator(modelFn=model_fn,
                       featureMapping={"features":"features"},
                       modelDir=model_dir,
                       labelCol="rating",
                       batchSize=128,
                       maxSteps=20000,
                       isValidationCol="isVal",  
                       modelFnParams={"hidden_layer1": 30, "hidden_layer2": 20, "learning_rate": 0.0001},
                       saveCheckpointsSecs=30)
transformer = est.fit(trainValDF)

# COMMAND ----------

predDF = transformer.transform(concatTestDF)
display(predDF.select("userId", "movieId", "predictions", "rating"))

# COMMAND ----------

from pyspark.sql.types import FloatType
def _pred(v):
  return float(v[0])

pred = udf(_pred, FloatType())
predDF = predDF.withColumn("prediction", pred(predDF.predictions))

# COMMAND ----------

testMse = regEval.evaluate(predDF)

print('The model had a MSE on the test set of {0}'.format(testMse))