# Databricks notebook source
spark.conf.set(
  "fs.azure.account.key.dewsa.blob.core.windows.net",
  "TDjjpKGq75D4t0fy4D5fNTJrnzlrtCKxB9MIipsfQvLWsR3mScAswOZTDOMi0JSXi1fb2ga5ly4rUAQ+WUvETQ==")

# COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://cat@dewsa.blob.core.windows.net/",
  mount_point = "/mnt/data",
  extra_configs = {"fs.azure.account.key.dewsa.blob.core.windows.net": "TDjjpKGq75D4t0fy4D5fNTJrnzlrtCKxB9MIipsfQvLWsR3mScAswOZTDOMi0JSXi1fb2ga5ly4rUAQ+WUvETQ=="})

# COMMAND ----------

# MAGIC %fs ls /mnt/data

# COMMAND ----------

df = spark.read.csv('dbfs:/mnt/data/MSFTTextDataAnalytics.csv', header='true', inferSchema = 'true')

# COMMAND ----------

df.show(5)

# COMMAND ----------

df.schema.names

# COMMAND ----------


oldColumns = df.schema.names
newColumns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Range", "Volume", "Pattern", "5DayStoch", "AvgStoch", "5DaySMA", "13daySMA", "Zone", "Reccomentation"]

df = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], newColumns[idx]), xrange(len(oldColumns)), df)
df.printSchema()
df.show(5)

# COMMAND ----------

df.createOrReplaceTempView("stock")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from stock limit 10

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.columns

# COMMAND ----------

df.show(5)

# COMMAND ----------

from pyspark.sql.functions import *
df = df.withColumn('Recommendation',when(df.Reccomentation == "Short",0).otherwise(1))

# COMMAND ----------

df.show(5)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
stages = [] # stages in our Pipeline


# COMMAND ----------

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol = "Recommendation", outputCol = "label")
stages += [label_stringIdx]

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
numericCols = ["Open", "High", "Low", "Close", "Range", "Volume", "5DayStoch", "AvgStoch", "5DaySMA", "13daySMA", "Zone", "Recommendation"]
assemblerInputs = numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

df.show(5)

# COMMAND ----------

dataset = df
# Create a Pipeline.
pipeline = Pipeline(stages=stages)
# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
pipelineModel = pipeline.fit(dataset)
dataset = pipelineModel.transform(dataset)

# Keep relevant columns
selectedcols = ["label", "features"]
dataset = dataset.select(selectedcols)
display(dataset)

# COMMAND ----------

### Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print trainingData.count()
print testData.count()

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
# You can select any columns in the above schema to view as well. For example's sake we will choose age & occupation
selected = predictions.select("label", "prediction", "probability", "features", "rawPrediction")
display(selected)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

evaluator.

# COMMAND ----------

print lr.explainParams()

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(trainingData)
# this will likely take a fair amount of time because of the amount of models that we're creating and testing

# COMMAND ----------

# Use test set here so we can measure the accuracy of our model on new data
predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

print 'Model Intercept: ', cvModel.bestModel.intercept

# COMMAND ----------

print(cvModel.bestModel)

# COMMAND ----------

weights = cvModel.bestModel.weights
# on Spark 2.X weights are available as ceofficients
# weights = cvModel.bestModel.coefficients
weights = map(lambda w: (float(w),), weights)  # convert numpy type to float, and to tuple
weightsDF = sqlContext.createDataFrame(weights, ["Feature Weight"])
display(weightsDF)

# COMMAND ----------

# View best model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "features", "rawPrediction")
display(selected)