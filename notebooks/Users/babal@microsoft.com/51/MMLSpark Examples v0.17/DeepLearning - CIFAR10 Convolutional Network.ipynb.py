# Databricks notebook source
# MAGIC %md ## 301 - Ingesting CIFAR Images into Spark DataFrames and Evaluating Pre-Trained CNTK Models

# COMMAND ----------

from mmlspark import CNTKModel, ModelDownloader
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from os.path import abspath

# COMMAND ----------

# MAGIC %md Set some paths.

# COMMAND ----------

cdnURL = "https://mmlspark.azureedge.net/datasets"

# Please note that this is a copy of the CIFAR10 dataset originally found here:
# http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
dataFile = "cifar-10-python.tar.gz"
dataURL = cdnURL + "/CIFAR10/" + dataFile

# COMMAND ----------

modelName = "ConvNet"
modelDir = "dbfs:///models/"

# COMMAND ----------

# MAGIC %md Get the model and extract the data.

# COMMAND ----------

import os, tarfile, pickle
import urllib.request

d = ModelDownloader(spark, modelDir)
model = d.downloadByName(modelName)
if not os.path.isfile(dataFile):
    urllib.request.urlretrieve(dataURL, dataFile)
with tarfile.open(dataFile, "r:gz") as f:
    test_dict = pickle.load(f.extractfile("cifar-10-batches-py/test_batch"),
                            encoding="latin1")

# COMMAND ----------

# MAGIC %md Preprocess the images.

# COMMAND ----------

import array
from pyspark.sql.functions import col
from pyspark.sql.types import *

def reshape_image(record):
    image, label, filename = record
    data = [float(x) for x in image.reshape(3,32,32).flatten()]
    return data, label, filename

convert_to_float = udf(lambda x: x, ArrayType(FloatType()))

image_rdd = zip(test_dict["data"], test_dict["labels"], test_dict["filenames"])
image_rdd = spark.sparkContext.parallelize(image_rdd).map(reshape_image)

imagesWithLabels = image_rdd.toDF(["images", "labels", "filename"])
imagesWithLabels = imagesWithLabels.withColumn("images", convert_to_float(col("images")))
imagesWithLabels.printSchema()

imagesWithLabels.cache()

# COMMAND ----------

# MAGIC %md Evaluate CNTK model.

# COMMAND ----------

import time
start = time.time()

# Use CNTK model to get log probabilities
cntkModel = CNTKModel().setInputCol("images").setOutputCol("output") \
                       .setModelLocation(model.uri).setOutputNode("z")
scoredImages = cntkModel.transform(imagesWithLabels)

# Transform the log probabilities to predictions
def argmax(x): return max(enumerate(x),key=lambda p: p[1])[0]

argmaxUDF = udf(argmax, IntegerType())
imagePredictions = scoredImages.withColumn("predictions", argmaxUDF("output")) \
                               .select("predictions", "labels")

numRows = imagePredictions.count()

end = time.time()
print("classifying {} images took {} seconds".format(numRows,end-start))

# COMMAND ----------

# MAGIC %md Plot confusion matrix.

# COMMAND ----------

imagePredictions = imagePredictions.toPandas()
y, y_hat = imagePredictions["labels"], imagePredictions["predictions"]

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_hat)

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
          "horse", "ship", "truck"]
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=90)
plt.yticks(tick_marks, labels)
plt.xlabel("Predicted label")
plt.ylabel("True Label")
display(plt.show())