# Databricks notebook source
# MAGIC %md 
# MAGIC # Distributed model inference using TensorFlow
# MAGIC This notebook demonstrates how to do distributed model inference using TensorFlow with ResNet-50 model and TFRecords as input data. 
# MAGIC 
# MAGIC This guide consists of the following sections:
# MAGIC 
# MAGIC * **Prepare trained model and data for inference.**
# MAGIC   * Downloads pre-trained versions of [ResNet-50](http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW.tar.gz).
# MAGIC   * Downloads the flowers data, uncompresses it, reads the files that make up the flowers data and creates TFRecord datasets.
# MAGIC * **Load the data into Spark DataFrames.** 
# MAGIC * **Run model inference via Pandas UDF.** 
# MAGIC 
# MAGIC **Note:**
# MAGIC * The notebook runs without code changes on CPU or GPU-enabled Apache Spark clusters.
# MAGIC * To run the notebook, create a cluster with Databricks Runtime 5.0 ML or above and `pyarrow==0.10.0`.

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "5000") # Please set a large batch size in practice.

# COMMAND ----------

import numpy as np
import os
import pandas as pd
import shutil
import tarfile
import time
import zipfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
  
from pyspark.sql.functions import col, pandas_udf, PandasUDFType

import tensorflow as tf
tf.__version__

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Prepare trained model and data for inference

# COMMAND ----------

# MAGIC %md
# MAGIC Define the variables and function to download the ResNet model.

# COMMAND ----------

# Internet URL for the tar-file with the Inception model.
# Note that this might change in the future and will need to be updated.
data_url = "http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW.tar.gz"
# Directory to store the downloaded data.
data_dir = "resnet/"
tensorflow_graph_dir = data_dir + 'resnet_v2_fp32_savedmodel_NCHW/1538687196/'

# COMMAND ----------

def maybe_download_and_extract(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)
    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        file_path, _ = urlretrieve(url=url, filename=file_path)

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")

# COMMAND ----------

# MAGIC %md
# MAGIC Download the ResNet model.

# COMMAND ----------

maybe_download_and_extract(url=data_url, download_dir=data_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC Load and broadcast the model.

# COMMAND ----------

sess = tf.Session()
model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], tensorflow_graph_dir)
bc_model = sc.broadcast(model)
sess.close()

# COMMAND ----------

# MAGIC %md
# MAGIC Run the script from [TensorFlow/Model](https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_flowers.py) to download the Flowers data and convert to TFRecords. It might take few minutes to download the data.

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://github.com/tensorflow/models/
# MAGIC cd models/research/slim/
# MAGIC DATA_DIR=/tmp/data/flowers
# MAGIC /databricks/python/bin/python download_and_convert_data.py \
# MAGIC     --dataset_name=flowers \
# MAGIC     --dataset_dir="${DATA_DIR}"
# MAGIC cp -r /tmp/data/flowers /dbfs/tmp

# COMMAND ----------

input_local_dir = "/tmp/flowers"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the data into Spark DataFrames
# MAGIC Use `spark-tensorflow-connector` to load TFRecords to Spark DataFrames.

# COMMAND ----------

from pyspark.sql.types import *

schema = StructType([StructField('image/class/label', IntegerType(), True),
                     StructField('image/width', IntegerType(), True),
                     StructField('image/height', IntegerType(), True),
                     StructField('image/format', StringType(), True),
                     StructField('image/encoded', BinaryType(), True)])

df = spark.read.format("tfrecords").schema(schema).load(input_local_dir+'/flowers_train*.tfrecord')
df = df.limit(3200)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run model inference via Pandas UDF

# COMMAND ----------

# MAGIC %md
# MAGIC Define the function to parse the TFRecords.

# COMMAND ----------

def parse_example(image_data):

  image = tf.image.decode_jpeg(image_data, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.central_crop(image, central_fraction=0.875)
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [224, 224],
                                     align_corners=False)
  image = tf.squeeze(image, [0])
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


# COMMAND ----------

# MAGIC %md
# MAGIC Define the function for model inference. 

# COMMAND ----------

def predict_batch(image_batch):
  batch_size = len(image_batch)
  sess = tf.Session()

  batch_size = 64
  image_input = tf.placeholder(dtype=tf.string, shape=[None])
  dataset = tf.data.Dataset.from_tensor_slices(image_input)
  dataset = dataset.map(parse_example, num_parallel_calls=16).prefetch(512).batch(batch_size)
  iterator = dataset.make_initializable_iterator()
  image = iterator.get_next()

  tf.train.import_meta_graph(bc_model.value)
  sess.run(tf.global_variables_initializer())
  sess.run(iterator.initializer, feed_dict={image_input: image_batch})
  softmax_tensor = sess.graph.get_tensor_by_name('softmax_tensor:0')
  result = []
  try:
    while True: 
      batch = sess.run(image)
      preds = sess.run(softmax_tensor, {'input_tensor:0': batch})
      result = result + list(preds)
  except tf.errors.OutOfRangeError:
    pass

  return pd.Series(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Test the function locally.

# COMMAND ----------

image_batch = df.limit(128).toPandas().loc[: , "image/encoded"].apply(lambda x: bytes(x))
images = predict_batch(image_batch)
print(images.shape)

# COMMAND ----------

# MAGIC %md 
# MAGIC Wrap the function as a Pandas UDF.

# COMMAND ----------

predict_batch_udf = pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)(predict_batch)
predictions = df.select(predict_batch_udf(col("image/encoded")).alias("prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the model inference and save the result to a Parquet file.

# COMMAND ----------

predictions.write.mode("overwrite").save("/tmp/predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC Load and check the prediction results.

# COMMAND ----------

result_df = spark.read.load("/tmp/predictions")
display(result_df)