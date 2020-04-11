# Databricks notebook source
# MAGIC %md ### News Recommendation ALS Example Databricks Notebook
# MAGIC ##### by Daniel Ciborowski, dciborow@microsoft.com
# MAGIC 
# MAGIC ##### Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC ##### Licensed under the MIT License.
# MAGIC 
# MAGIC ##### Setup
# MAGIC 1. Create new Cluster, DB 4.1, Spark 2.3.0, Python3
# MAGIC 1. (Optional for Ranking Metrics) From Maven add to cluster the following jar: Azure:mmlspark:0.15

# COMMAND ----------

import pandas as pd
import random

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.sql.functions import col, collect_list

# COMMAND ----------

# Create Sample Data
raw = [
  {'userId': 1, 'itemId': 1, 'rating':  random.randint(0, 10)},
  {'userId': 2, 'itemId': 1, 'rating':  random.randint(0, 10)},
  {'userId': 3, 'itemId': 1, 'rating':  random.randint(0, 10)},
  {'userId': 4, 'itemId': 1, 'rating':  random.randint(0, 10)},
  {'userId': 5, 'itemId': 1, 'rating':  random.randint(0, 10)},
  {'userId': 1, 'itemId': 2, 'rating':  random.randint(0, 10)},
  {'userId': 2, 'itemId': 2, 'rating':  random.randint(0, 10)},
  {'userId': 3, 'itemId': 2, 'rating':  random.randint(0, 10)},
  {'userId': 4, 'itemId': 2, 'rating':  random.randint(0, 10)},
  {'userId': 5, 'itemId': 2, 'rating':  random.randint(0, 10)},
  {'userId': 1, 'itemId': 3, 'rating':  random.randint(0, 10)},
  {'userId': 2, 'itemId': 3, 'rating':  random.randint(0, 10)},
  {'userId': 3, 'itemId': 3, 'rating':  random.randint(0, 10)},
  {'userId': 4, 'itemId': 3, 'rating':  random.randint(0, 10)},
  {'userId': 5, 'itemId': 3, 'rating':  random.randint(0, 10)},
  {'userId': 1, 'itemId': 4, 'rating':  random.randint(0, 10)},
  {'userId': 2, 'itemId': 4, 'rating':  random.randint(0, 10)},
  {'userId': 3, 'itemId': 4, 'rating':  random.randint(0, 10)},
  {'userId': 4, 'itemId': 4, 'rating':  random.randint(0, 10)},
  {'userId': 5, 'itemId': 4, 'rating':  random.randint(0, 10)},  
  {'userId': 1, 'itemId': 5, 'rating':  random.randint(0, 10)},
  {'userId': 2, 'itemId': 5, 'rating':  random.randint(0, 10)},
  {'userId': 3, 'itemId': 5, 'rating':  random.randint(0, 10)},
  {'userId': 4, 'itemId': 5, 'rating':  random.randint(0, 10)},
  {'userId': 5, 'itemId': 5, 'rating':  random.randint(0, 10)},   
]

day1 = pd.DataFrame(raw)
day2=pd.DataFrame(raw)
day2['itemId'] = day2['itemId']+10
day3=pd.DataFrame(raw)
day3['itemId'] = day3['itemId']+20
day4=pd.DataFrame(raw)
day4['itemId'] = day4['itemId']+30

data = day1 \
  .append(day2) \
  .append(day3) \
  .append(day4) \
  .sample(frac=0.75, replace=False)

spark = SparkSession.builder.getOrCreate()
ratings = spark.createDataFrame(data)
display(ratings.select('userId','itemId','rating').orderBy('userId','itemId'))

# COMMAND ----------

# Build the recommendation model using ALS on the rating data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
algo = ALS(userCol="userId", itemCol="itemId", implicitPrefs=True, coldStartStrategy="drop")
model = algo.fit(ratings)

# COMMAND ----------

# Evaluate the model by computing the RMSE on the rating data
predictions = model.transform(ratings)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

# Evaluate the model by computing ranking metrics on the rating data
from mmlspark.RankingAdapter import RankingAdapter
from mmlspark.RankingEvaluator import RankingEvaluator

output = RankingAdapter(mode='allUsers', k=5, recommender=algo) \
  .fit(ratings) \
  .transform(ratings)

metrics = ['ndcgAt','map','recallAtK','mrr','fcp']
metrics_dict = {}
for metric in metrics:
    metrics_dict[metric] = RankingEvaluator(k=3, metricName=metric).evaluate(output)
    
metrics_dict    

# COMMAND ----------

# Recommend Subset Wrapper
def recommendSubset(self, df):
  def Func(lines):
    out = []
    for i in range(len(lines[1])):
      out += [(lines[1][i],lines[2][i])]
    return lines[0], out

  tup = StructType([
    StructField('itemId', IntegerType(), True),
    StructField('rating', FloatType(), True)
  ])
  array_type = ArrayType(tup, True)

  scoring = spark.createDataFrame(day4)
  scored = self.transform(scoring)

  recs = scored \
    .groupBy(col('userId')) \
    .agg(collect_list(col("itemId")),collect_list(col("prediction"))) \
    .rdd \
    .map(Func) \
    .toDF() \
    .withColumnRenamed("_1","userId") \
    .withColumnRenamed("_2","recommendations") \
    .select(col("userId"),col("recommendations").cast(array_type))

  return recs

import pyspark
pyspark.ml.recommendation.ALSModel.recommendSubset = recommendSubset

# COMMAND ----------

# Recommend most recent items for all users
day4df = spark.createDataFrame(day4)
recs = model.recommendSubset(scoring)

display(recs.orderBy('userId'))

# COMMAND ----------

