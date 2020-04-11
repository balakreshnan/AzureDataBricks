// Databricks notebook source
// MAGIC %md
// MAGIC #Author
// MAGIC <a href="https://www.linkedin.com/in/yshkurnykov">Shkurnykov  Yevhen</a>&nbsp; for group <a href="https://www.facebook.com/groups/bigdatashuffle/">Big Data Shuffle</a>

// COMMAND ----------

// MAGIC %md
// MAGIC Titanic: Machine Learning from Disaster (Kaggle)
// MAGIC --------------------------------------
// MAGIC <a href="https://www.kaggle.com/c/titanic/data">Competition on Kaggle</a><br/>
// MAGIC <a href="https://benfradet.github.io/blog/2015/12/16/Exploring-spark.ml-with-the-Titanic-Kaggle-competition">Source</a>

// COMMAND ----------

// MAGIC %md
// MAGIC <h3>Data Dictionary</h3>
// MAGIC <table style="width: 100%;">
// MAGIC <tbody>
// MAGIC <tr><th><b>Variable</b></th><th><b>Definition</b></th><th><b>Key</b></th></tr>
// MAGIC <tr>
// MAGIC <td>survival</td>
// MAGIC <td>Survival</td>
// MAGIC <td>0 = No, 1 = Yes</td>
// MAGIC </tr>
// MAGIC <tr>
// MAGIC <td>pclass</td>
// MAGIC <td>Ticket class</td>
// MAGIC <td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
// MAGIC </tr>
// MAGIC <tr>
// MAGIC <td>sex</td>
// MAGIC <td>Sex</td>
// MAGIC <td></td>
// MAGIC </tr>
// MAGIC <tr>
// MAGIC <td>Age</td>
// MAGIC <td>Age in years</td>
// MAGIC <td></td>
// MAGIC </tr>
// MAGIC <tr>
// MAGIC <td>sibsp</td>
// MAGIC <td># of siblings / spouses aboard the Titanic</td>
// MAGIC <td></td>
// MAGIC </tr>
// MAGIC <tr>
// MAGIC <td>parch</td>
// MAGIC <td># of parents / children aboard the Titanic</td>
// MAGIC <td></td>
// MAGIC </tr>
// MAGIC <tr>
// MAGIC <td>ticket</td>
// MAGIC <td>Ticket number</td>
// MAGIC <td></td>
// MAGIC </tr>
// MAGIC <tr>
// MAGIC <td>fare</td>
// MAGIC <td>Passenger fare</td>
// MAGIC <td></td>
// MAGIC </tr>
// MAGIC <tr>
// MAGIC <td>cabin</td>
// MAGIC <td>Cabin number</td>
// MAGIC <td></td>
// MAGIC </tr>
// MAGIC <tr>
// MAGIC <td>embarked</td>
// MAGIC <td>Port of Embarkation</td>
// MAGIC <td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
// MAGIC </tr>
// MAGIC </tbody>
// MAGIC </table>
// MAGIC 
// MAGIC <h3>Variable Notes</h3>
// MAGIC <p><b>pclass</b>: A proxy for socio-economic status (SES)<br> 1st = Upper<br> 2nd = Middle<br> 3rd = Lower<br><br> <b>age</b>: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5<br><br> <b>sibsp</b>: The dataset defines family relations in this way...<br> Sibling = brother, sister, stepbrother, stepsister<br> Spouse = husband, wife (mistresses and fiancés were ignored)<br><br> <b>parch</b>: The dataset defines family relations in this way...<br> Parent = mother, father<br> Child = daughter, son, stepdaughter, stepson<br> Some children travelled only with a nanny, therefore parch=0 for them.</p>

// COMMAND ----------

// MAGIC %md
// MAGIC #Import section

// COMMAND ----------

import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, FloatType};
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

// COMMAND ----------

// MAGIC %md
// MAGIC #Import files

// COMMAND ----------

// MAGIC %md
// MAGIC #Create scheme

// COMMAND ----------

// MAGIC %md
// MAGIC We will use only train set for learning and test

// COMMAND ----------

val dataScheme = (new StructType)
.add("PassengerId", IntegerType)
.add("Survived", IntegerType)
.add("Pclass", IntegerType)
.add("Name", StringType)
.add("Sex", StringType)
.add("Age", FloatType)
.add("SibSp", IntegerType)
.add("Parch", IntegerType)
.add("Ticket", StringType)
.add("Fare", FloatType)
.add("Cabin", StringType)
.add("Embarked", StringType)

// COMMAND ----------

val datasetDF=sqlContext.read.schema(dataScheme).option("header", "true").csv("/FileStore/tables/i31jf15l1496724300776/train.csv")
datasetDF.createOrReplaceTempView("train")

// COMMAND ----------

// MAGIC %sql select * from train

// COMMAND ----------

// MAGIC %md
// MAGIC #User defined function

// COMMAND ----------

val embarked: (String => String) = {
  case "" => "S"
  case null =>"S"
  case a  => a
}
val embarkedUDF = udf(embarked)

// COMMAND ----------

// MAGIC %md
// MAGIC #Calculate average age and fare for fill the gaps

// COMMAND ----------

//Calculate average age for filling gaps in dataset
val averageAge = datasetDF.select("Age")
  .agg(avg("Age"))
  .collect() match {
  case Array(Row(avg: Double)) => avg
  case _ => 0
}

//Calculate average fare for filling gaps in dataset
val averageFare = datasetDF.select("Fare")
  .agg(avg("Fare"))
  .collect() match {
  case Array(Row(avg: Double)) => avg
  case _ => 0
}


// COMMAND ----------

// MAGIC %md
// MAGIC #Fill the gaps

// COMMAND ----------

val filledDF = datasetDF.na.fill(Map("Fare" -> averageFare, "Age" -> averageAge))
val filledDF2 = filledDF.withColumn("Embarked", embarkedUDF(filledDF.col("Embarked")))  
val Array(trainingData, testData) = filledDF2.randomSplit(Array(0.7, 0.3))

// COMMAND ----------

// MAGIC %md
// MAGIC #Indexing features

// COMMAND ----------

//Indexing categorical fetures
val featuresCatColNames = Seq("Pclass", "Sex", "Embarked")
val stringIndexers = featuresCatColNames.map { colName =>
  new StringIndexer()
    .setInputCol(colName)
    .setOutputCol(colName + "Indexed")
    .fit(trainingData)
}

//Indexing label
val labelIndexer = new StringIndexer()
.setInputCol("Survived")
.setOutputCol("SurvivedIndexed")
.fit(trainingData)

val featuresNumColNames = Seq("Age", "SibSp", "Parch", "Fare")
val indexedfeaturesCatColNames = featuresCatColNames.map(_ + "Indexed")
val allIndexedFeaturesColNames = featuresNumColNames ++ indexedfeaturesCatColNames
val assembler = new VectorAssembler()
  .setInputCols(Array(allIndexedFeaturesColNames: _*))
  .setOutputCol("Features")

// COMMAND ----------

// MAGIC %md
// MAGIC #Create classifier

// COMMAND ----------

val randomForest = new RandomForestClassifier()
  .setLabelCol("SurvivedIndexed")
  .setFeaturesCol("Features")

//Retrieve original labels
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// COMMAND ----------

// define the order of the operations to be performed
val pipeline = new Pipeline().setStages(
  (stringIndexers :+ labelIndexer :+ assembler :+ randomForest :+ labelConverter).toArray)


// COMMAND ----------

// MAGIC %md
// MAGIC #Cross validation

// COMMAND ----------

 // grid of values to perform cross validation on
 val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxBins, Array(25, 28, 31))
      .addGrid(randomForest.maxDepth, Array(4, 6, 8))
      .addGrid(randomForest.impurity, Array("entropy", "gini"))
      .build()

val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("SurvivedIndexed")

val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    // train the model
val crossValidatorModel = cv.fit(trainingData)

// COMMAND ----------

// MAGIC %md
// MAGIC #Predictions and test accuracy

// COMMAND ----------

// make predictions
val predictions = crossValidatorModel.transform(testData)

//Accuracy
val accuracy = evaluator.evaluate(predictions)
println("Test Error DT= " + (1.0 - accuracy))