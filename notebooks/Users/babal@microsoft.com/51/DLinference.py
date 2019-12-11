# Databricks notebook source
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import horovod.tensorflow as hvd

from pyspark.sql.types import *
from pyspark.sql.functions import rand, when

from sparkdl.estimators.horovod_estimator.estimator import HorovodEstimator

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/horovod_estimator/"))

# COMMAND ----------

# Load MNIST dataset, with images represented as arrays of floats
mnist_dataset = learn.datasets.mnist.load_mnist("/tmp/sid/tfrecords")
train_images, train_labels = mnist_dataset.train.images, mnist_dataset.train.labels
data = [(train_images[i].tolist(), int(train_labels[i])) for i in range(len(train_images))]
schema = StructType([StructField("image", ArrayType(FloatType())),
                     StructField("label_col", LongType())])
df = spark.createDataFrame(data, schema)

# COMMAND ----------

display(df)

# COMMAND ----------

dbutils.fs.ls("file:/tmp/horovod_saved/")

# COMMAND ----------

tensorflow_graph_dir = '/tmp/horovod_saved'

# COMMAND ----------

sess = tf.Session()
model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], tensorflow_graph_dir)
bc_model = sc.broadcast(model)
sess.close()

# COMMAND ----------

