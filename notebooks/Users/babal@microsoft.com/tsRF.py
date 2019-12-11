# Databricks notebook source
df = spark.read.csv('dbfs:/mnt/data/MSFTTextDataAnalytics.csv', header='true', inferSchema = 'true')

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

df.columns

# COMMAND ----------

from pyspark.sql.functions import *
df = df.withColumn('Recommendation',when(df.Reccomentation == "Short",0).otherwise(1))

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

dataset.show(5)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

data = dataset
# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only

# COMMAND ----------

predictions.show()

# COMMAND ----------

predictions.schema

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
evaluator.evaluate(predictions)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

print(evaluator)

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())

# COMMAND ----------

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [1 10, 20])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())

# COMMAND ----------

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=10)

cvModel = crossval.fit(trainingData)

# COMMAND ----------

cvpredictions = cvModel.transform(testData)

# COMMAND ----------

evaluator.evaluate(cvpredictions)

# COMMAND ----------

cvpredictions.schema

# COMMAND ----------

selected = cvpredictions.select("label", "prediction", "probability", "predictedLabel", "features")
display(selected)

# COMMAND ----------

bestModel = cvModel.bestModel

# COMMAND ----------

# Generate predictions for entire dataset
finalPredictions = bestModel.transform(dataset)

# COMMAND ----------

# Evaluate best model
evaluator.evaluate(finalPredictions)

# COMMAND ----------

finalPredictions.createOrReplaceTempView("finalPredictions")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM finalPredictions

# COMMAND ----------

# MAGIC %sql
# MAGIC select label, prediction from finalPredictions limit 400