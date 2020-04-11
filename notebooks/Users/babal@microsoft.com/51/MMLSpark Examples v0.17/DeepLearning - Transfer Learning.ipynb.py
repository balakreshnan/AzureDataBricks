# Databricks notebook source
# MAGIC %md ## 303 - Transfer Learning by DNN Featurization
# MAGIC 
# MAGIC Classify automobile vs airplane using DNN featurization and transfer learning
# MAGIC against a subset of images from CIFAR-10 dataset.

# COMMAND ----------

# MAGIC %md First, we load first batch of CIFAR-10 training data into NumPy array.

# COMMAND ----------

from mmlspark import CNTKModel, ModelDownloader
import numpy as np, pandas as pd
import os, urllib, tarfile, pickle, array
from os.path import abspath
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *

cdnURL = "https://mmlspark.azureedge.net/datasets"

# Please note that this is a copy of the CIFAR10 dataset originally found here:
# http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
dataFile = "cifar-10-python.tar.gz"
dataURL = cdnURL + "/CIFAR10/" + dataFile

if not os.path.isfile(dataFile):
    urllib.request.urlretrieve(dataURL, dataFile)
with tarfile.open(dataFile, "r:gz") as f:
    train_dict = pickle.load(f.extractfile("cifar-10-batches-py/data_batch_1"),
                             encoding="latin1")

train_data = np.array(train_dict["data"])
train_labels = np.array(train_dict["labels"])

# COMMAND ----------

# MAGIC %md Load DNN Model and pick one of the inner layers as feature output

# COMMAND ----------

modelName = "ConvNet"
modelDir = "wasb:///models/"
modelDir = "file:" + abspath("models")
d = ModelDownloader(spark, modelDir)
model = d.downloadByName(modelName)
print(model.layerNames)
cntkModel = CNTKModel().setInputCol("images").setOutputCol("features") \
                       .setModelLocation(model.uri).setOutputNode("l8")

# COMMAND ----------

# MAGIC %md Format raw CIFAR data into correct shape.

# COMMAND ----------

def reshape_image(record):
    image, label = record
    data = [float(x) for x in image.reshape(3,32,32).flatten()]
    return data, int(label)

convert_to_float = udf(lambda x: x, ArrayType(FloatType()))

image_rdd = zip(train_data,train_labels)
image_rdd = spark.sparkContext.parallelize(image_rdd).map(reshape_image)

imagesWithLabels = image_rdd.toDF(["images", "labels"])
imagesWithLabels = imagesWithLabels.withColumn("images", convert_to_float(col("images")))

# COMMAND ----------

# MAGIC %md Select airplanes (label=0) and automobiles (label=1)

# COMMAND ----------

imagesWithLabels = imagesWithLabels.filter("labels<2")
imagesWithLabels.cache()

# COMMAND ----------

# MAGIC %md Featurize images

# COMMAND ----------

featurizedImages = cntkModel.transform(imagesWithLabels).select(["features","labels"])

# COMMAND ----------

# MAGIC %md Use featurized images to train a classifier

# COMMAND ----------

from mmlspark import TrainClassifier
from pyspark.ml.classification import RandomForestClassifier

train,test = featurizedImages.randomSplit([0.75,0.25])

model = TrainClassifier(model=RandomForestClassifier(),labelCol="labels").fit(train)

# COMMAND ----------

# MAGIC %md Evaluate the accuracy of the model

# COMMAND ----------

from mmlspark import ComputeModelStatistics
predictions = model.transform(test)
metrics = ComputeModelStatistics(evaluationMetric="accuracy").transform(predictions)
metrics.show()