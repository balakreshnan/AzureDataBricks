# Databricks notebook source
# MAGIC %md ## 101 - Training and Evaluating Classifiers with `mmlspark`
# MAGIC 
# MAGIC In this example, we try to predict incomes from the *Adult Census* dataset.
# MAGIC 
# MAGIC First, we import the packages (use `help(mmlspark)` to view contents),

# COMMAND ----------

import numpy as np
import pandas as pd
import mmlspark

# help(mmlspark)

# COMMAND ----------

# MAGIC %md Now let's read the data and split it to train and test sets:

# COMMAND ----------

dataFilePath = "AdultCensusIncome.csv"
import os, urllib
if not os.path.isfile(dataFilePath):
    urllib.request.urlretrieve("https://mmlspark.azureedge.net/datasets/" + dataFilePath, dataFilePath)
data = spark.createDataFrame(pd.read_csv(dataFilePath, dtype={" hours-per-week": np.float64}))
data = data.select([" education", " marital-status", " hours-per-week", " income"])
train, test = data.randomSplit([0.75, 0.25], seed=123)
train.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md `TrainClassifier` can be used to initialize and fit a model, it wraps SparkML classifiers.
# MAGIC You can use `help(mmlspark.TrainClassifier)` to view the different parameters.
# MAGIC 
# MAGIC Note that it implicitly converts the data into the format expected by the algorithm: tokenize
# MAGIC and hash strings, one-hot encodes categorical variables, assembles the features into a vector
# MAGIC and so on.  The parameter `numFeatures` controls the number of hashed features.

# COMMAND ----------

from mmlspark import TrainClassifier
from pyspark.ml.classification import LogisticRegression
model = TrainClassifier(model=LogisticRegression(), labelCol=" income", numFeatures=256).fit(train)
model.write().overwrite().save("adultCensusIncomeModel.mml")

# COMMAND ----------

# MAGIC %md After the model is trained, we score it against the test dataset and view metrics.

# COMMAND ----------

from mmlspark import ComputeModelStatistics, TrainedClassifierModel
predictionModel = TrainedClassifierModel.load("adultCensusIncomeModel.mml")
prediction = predictionModel.transform(test)
metrics = ComputeModelStatistics().transform(prediction)
metrics.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md Finally, we save the model so it can be used in a scoring program.

# COMMAND ----------

model.write().overwrite().save("AdultCensus.mml")