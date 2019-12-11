# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed model inference using Keras
# MAGIC This notebook demonstrates how to do distributed model inference using Keras with ResNet-50 model and parquet file as input data.
# MAGIC 
# MAGIC This guide consists of the following sections:
# MAGIC 
# MAGIC * **Prepare trained model and data for inference.**
# MAGIC   * Load pre-trained ResNet-50 model from [keras.applications](https://keras.io/applications/).
# MAGIC   * Downloads the Flowers data and save to parquet files.
# MAGIC * **Load the data into Spark DataFrames.** 
# MAGIC * **Run model inference via Pandas UDF.** 
# MAGIC 
# MAGIC **Note:**
# MAGIC * The notebook runs without code changes on CPU or GPU-enabled Apache Spark clusters.
# MAGIC * To run the notebook, create a cluster with Databricks Runtime 5.0 ML or above.

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1600") # Please set a large batch size in practice.

# COMMAND ----------

import os
import shutil
import time
import pandas as pd
from PIL import Image
import numpy as np

import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50

from pyspark.sql.functions import col, pandas_udf, PandasUDFType

# COMMAND ----------

# MAGIC %md
# MAGIC Define the input and output directory.

# COMMAND ----------

file_name = "image_data.parquet"
dbfs_file_path = "/dbfs/tmp/flowers/image_data.parquet"
local_file_path = "/tmp/flowers/image_data.parquet"
output_file_path = "/tmp/predictions"

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Prepare trained model and data for inference

# COMMAND ----------

# MAGIC %md
# MAGIC Load the ResNet-50 Model and broadcast the weights.

# COMMAND ----------

model = ResNet50()
bc_model_weights = sc.broadcast(model.get_weights())

# COMMAND ----------

# MAGIC %md
# MAGIC Download the Google flowers dataset.

# COMMAND ----------

# MAGIC %sh 
# MAGIC curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
# MAGIC tar xzf flower_photos.tgz &>/dev/null

# COMMAND ----------

import os
local_dir = "./flower_photos"
files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(local_dir) for f in filenames if os.path.splitext(f)[1] == '.jpg']
files

# COMMAND ----------

# MAGIC %md
# MAGIC Save the datasets to one parquet file.

# COMMAND ----------

image_data = []
for file in files:
  img = Image.open(file)
  img = img.resize([224, 224])
  data = np.asarray( img, dtype="float32" ).reshape([224*224*3])
  
  image_data.append({"data": data})

pandas_df = pd.DataFrame(image_data, columns = ['data'])
pandas_df.to_parquet(file_name)
shutil.copyfile(file_name, dbfs_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the data into Spark DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC Load the data in Spark

# COMMAND ----------

from pyspark.sql.types import *
df = spark.read.format("parquet").load(local_file_path)
print(df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run model inference via Pandas UDF

# COMMAND ----------

# MAGIC %md
# MAGIC Define the function to parse the input data.

# COMMAND ----------

def parse_image(image_data):
  image = tf.image.convert_image_dtype(image_data, dtype=tf.float32) * (2. / 255) - 1
  image = tf.reshape(image,[224,224,3])
  return image

# COMMAND ----------

# MAGIC %md
# MAGIC Define the function for model inference.

# COMMAND ----------

def predict_batch(image_batch):
  image_batch = np.stack(image_batch)
  batch_size = 64
  sess = tf.Session()
  
  model = ResNet50(weights=None)
  model.set_weights(bc_model_weights.value)
  
  image_input = tf.placeholder(dtype=tf.int32, shape=[None,224*224*3])
  dataset = tf.data.Dataset.from_tensor_slices(image_input)
  dataset = dataset.map(parse_image, num_parallel_calls=8).prefetch(5000).batch(batch_size)
  iterator = dataset.make_initializable_iterator()
  images_tensor = iterator.get_next()
  
  with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={image_input: image_batch})
    result = []
    try:
      while True:
        images = sess.run(images_tensor)
        preds = model.predict(images, steps=1)
        result = result + list(preds)
    except tf.errors.OutOfRangeError:
      pass

  return pd.Series(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Test the function locally.

# COMMAND ----------

features_batch = df.limit(128).toPandas().loc[: , "data"]
images = predict_batch(features_batch)
print(images.shape)

# COMMAND ----------

# MAGIC %md 
# MAGIC Wrap the function as a Pandas UDF.

# COMMAND ----------

predict_batch_udf = pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)(predict_batch)


# COMMAND ----------

# MAGIC %md
# MAGIC Run the model inference and save the result to parquet file.

# COMMAND ----------

predictions_df = df.select(predict_batch_udf(col("data")).alias("prediction"))
predictions_df.write.mode("overwrite").parquet(output_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can load and check the prediction results.

# COMMAND ----------

result_df = spark.read.load(output_file_path)
display(result_df)