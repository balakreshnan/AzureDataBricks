# Databricks notebook source
# MAGIC %md-sandbox 
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/cs110x/movie-camera.png" style="float:right; height: 200px; margin: 10px; border: 1px solid #ddd; border-radius: 15px 15px 15px 15px; padding: 10px"/>
# MAGIC 
# MAGIC # Scalable Movie Recommendations
# MAGIC 
# MAGIC In this notebook, we will cover how to build a highly scalable and accurate neural network using Spark and Horovod! 
# MAGIC 
# MAGIC To run this notebook, you must be using **Databricks 4.1 ML Beta** (includes Apache Spark 2.3.0, Scala 2.11) and Python 3. You will need one driver, and two or more workers.  
# MAGIC 
# MAGIC We will use 20 million movie ratings from the [MovieLens stable benchmark rating dataset](http://grouplens.org/datasets/movielens/) in this notebook.

# COMMAND ----------

movies_df_schema = "ID integer, title string"

moviesDF = spark.read.csv("mnt/databricks-datasets/cs110x/ml-20m/data-001/movies.csv", header=True, schema=movies_df_schema)

# COMMAND ----------

ratings_df_schema = "userId integer, movieId integer, rating float"

ratingsDF = spark.read.csv("mnt/databricks-datasets/cs110x/ml-20m/data-001/ratings.csv", header=True, schema=ratings_df_schema).cache()

ratingsCount = ratingsDF.count()
moviesCount = moviesDF.count()

print('There are {0} ratings and {1} movies in the datasets.'.format(ratingsCount, moviesCount))

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

spark.conf.set("spark.sql.shuffle.partitions", "16")

als = (ALS()
       .setUserCol("userId")
       .setItemCol("movieId")
       .setRatingCol("rating")
       .setPredictionCol("predictions")
       .setMaxIter(2)
       .setSeed(seed)
       .setRegParam(0.1)
       .setColdStartStrategy("drop")
       .setRank(12))

alsModel = als.fit(trainingDF)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regEval = RegressionEvaluator(predictionCol="predictions", labelCol="rating", metricName="mse")

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
                 .select('userId', 'movieId', concat_arrays_udf(col("iFeatures"), col("uFeatures")).alias("features"), "rating"))
concatTestDF = (joinedTestDF
                .select('userId', 'movieId', concat_arrays_udf(col("iFeatures"), col("uFeatures")).alias("features"), "rating"))

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

def model_fn(features, labels, mode, params):
    print("HVD Size: ", hvd.size())
    features_with_shape = tf.reshape(features["features"], [-1, 24]) # Explicitly specify dimensions
    
    hidden_layer1 = tf.layers.dense(inputs=features_with_shape, units=params["hidden_layer1"], activation=tf.nn.relu)
    hidden_layer2 = tf.layers.dense(inputs=hidden_layer1, units=params["hidden_layer2"], activation=tf.nn.relu)
    predictions = tf.squeeze(tf.layers.dense(inputs=hidden_layer2, units=1, activation=None), axis=-1)
    
    # If the estimator is running in PREDICT mode, we can stop building our model graph here and simply return
    # our model's inference outputs
    serving_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    export_outputs = {serving_key: tf.estimator.export.PredictOutput({"predictions": predictions})}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
      
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, predictions)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        optimizer = hvd.DistributedOptimizer(optimizer)
        
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          export_outputs=export_outputs)
    # If running in EVAL mode, add model evaluation metrics (accuracy) to our EstimatorSpec so that
    # they're logged when model evaluation runs
    eval_metric_ops = {"rmse": tf.metrics.root_mean_squared_error(labels=labels, predictions=predictions)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, export_outputs=export_outputs)

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

display(transformer.transform(concatTestDF).select("userId", "movieId", "predictions", "rating"))

# COMMAND ----------

testMse = regEval.evaluate(transformer.transform(concatTestDF))

print('The model had a MSE on the test set of {0}'.format(testMse))