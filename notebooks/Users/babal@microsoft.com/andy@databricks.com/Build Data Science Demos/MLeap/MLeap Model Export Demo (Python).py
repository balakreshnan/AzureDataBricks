# Databricks notebook source
# MAGIC %md #  Model Export with MLeap
# MAGIC 
# MAGIC MLeap is a common serialization format and execution engine for machine learning pipelines. It supports Apache Spark, Scikit-learn, and TensorFlow for training pipelines and exporting them to an MLeap Bundle. Serialized pipelines (bundles) can be deserialized back into Apache Spark, Scikit-learn, TensorFlow graphs, or an MLeap pipeline. This notebook demonstrates how to use MLeap to do the model export with MLlib. For an overview of the package and more examples, check out the [MLeap documentation](http://mleap-docs.combust.ml/).

# COMMAND ----------

# MAGIC %md ## Cluster Setup
# MAGIC  
# MAGIC To use MLeap's PySpark integration on the cluster: 
# MAGIC * Install MLeap Spark. Create a new library with the source option ``Maven Coordinate`` and Maven artifacts coordinate ``ml.combust.mleap:mleap-spark_2.11:0.10.1``. [Attach the library to a cluster](https://docs.databricks.com/user-guide/libraries.html)
# MAGIC * Install MLeap. Create a new library with the PyPI package named ``mleap``.
# MAGIC 
# MAGIC **Note: **
# MAGIC * This version of MLeap works with Spark 2.0, 2.1, 2.2, and 2.3. 
# MAGIC * It is cross-compiled only for Scala 2.11.

# COMMAND ----------

# MAGIC %md ## In this Notebook
# MAGIC 
# MAGIC This notebook demonstrates how to use MLeap to export a `DecisionTreeClassifier` from MLlib and how to load the saved PipelineModel to make predictions.
# MAGIC 
# MAGIC The basic workflow is as follows:
# MAGIC * Model export
# MAGIC   * Fit a PipelineModel using MLlib.
# MAGIC   * Use MLeap to serialize the PipelineModel to zip File or to directory.
# MAGIC * Move the PipelineModel files to your deployment project or data store.
# MAGIC * In your project
# MAGIC   * Use MLeap to deserialize the saved PipelineModel
# MAGIC   * Make predictions.

# COMMAND ----------

# MAGIC %md ## Training the Model by MLlib

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, Tokenizer, HashingTF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# MAGIC %fs ls /

# COMMAND ----------

input_df = spark.read.parquet("/databricks-datasets/news20.binary/data-001/training")

# COMMAND ----------

input_df.printSchema()

# COMMAND ----------

df = input_df.select("text", "topic")
df.cache()
display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md ### Define ML Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC WHAT IS A PIPELINE?
# MAGIC 
# MAGIC [[ data --> featurization (tokenize, index, tf) --> model (decision tree) ]]

# COMMAND ----------

labelIndexer = StringIndexer(inputCol="topic", outputCol="label", handleInvalid="keep")

# COMMAND ----------

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="features")

# COMMAND ----------

dt = DecisionTreeClassifier()
pipeline = Pipeline(stages=[labelIndexer, tokenizer, hashingTF, dt])

# COMMAND ----------

dir(hashingTF.numFeatures)

# COMMAND ----------

# MAGIC %md ### Tune ML Pipeline

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(hashingTF.numFeatures, [1000, 2000]).build()
cv = CrossValidator(estimator=pipeline, evaluator=MulticlassClassificationEvaluator(), estimatorParamMaps=paramGrid)

# COMMAND ----------

cvModel = cv.fit(df)

# COMMAND ----------

model = cvModel.bestModel

# COMMAND ----------

sparkTransformed = model.transform(df)
display(sparkTransformed)

# COMMAND ----------

# MAGIC %md ## Export Trained Model with MLeap

# COMMAND ----------

# MAGIC %md
# MAGIC MLeap supports serializing the model to one zip file. In order to serialize to a zip file, make sure the URI begins with ``jar:file`` and ends with a ``.zip``.

# COMMAND ----------

# MAGIC %md The supported transformers for MLeap can be found in the  [MLeap documentation](http://mleap-docs.combust.ml/core-concepts/transformers/support.html).

# COMMAND ----------

# MAGIC %sh 
# MAGIC rm -rf /tmp/mleap_python_model_export
# MAGIC mkdir /tmp/mleap_python_model_export

# COMMAND ----------

import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer

model.serializeToBundle("jar:file:/tmp/mleap_python_model_export/20news_pipeline-json.zip", sparkTransformed)

# COMMAND ----------

# MAGIC %md ## Download Model Files
# MAGIC In this example we download the model files from the browser. In general, you may want to programmatically move the model to a persistent storage layer.

# COMMAND ----------

# MAGIC %sh
# MAGIC cp /tmp/mleap_python_model_export/20news_pipeline-json.zip /dbfs/FileStore/20news_pipeline-json.zip

# COMMAND ----------

# MAGIC %md  Get a link to the downloadable zip via:
# MAGIC `https://[MY_DATABRICKS_URL]/files/[FILE_NAME].zip`.  E.g., if you access Databricks at `https://mycompany.databricks.com`, then your link would be:
# MAGIC `https://mycompany.databricks.com/files/20news_pipeline-json.zip`.

# COMMAND ----------

# MAGIC %md ## Import Trained Model with MLeap

# COMMAND ----------

# MAGIC %md 
# MAGIC This section shows how to use MLeap to load a trained model for use in your application. To use an existing ML models and pipelines to make predictions for new data, you can deserialize the model from the file you saved.

# COMMAND ----------

# MAGIC %md ### Import Model to PySpark
# MAGIC 
# MAGIC This section shows how to load an MLeap bundle and make predictions on a Spark DataFrame.  This can be useful if you want to use the same persistence format (bundle) for loading into Spark and non-Spark applications.  If your goal is to make predictions only in Spark, then we recommend using [MLlib's native ML persistence](https://spark.apache.org/docs/latest/ml-pipeline.html#ml-persistence-saving-and-loading-pipelines).

# COMMAND ----------

from pyspark.ml import PipelineModel
deserializedPipeline = PipelineModel.deserializeFromBundle("jar:file:/tmp/mleap_python_model_export/20news_pipeline-json.zip")

# COMMAND ----------

# MAGIC %md Now you can use the loaded model to make predictions.

# COMMAND ----------

test_df = spark.read.parquet("/databricks-datasets/news20.binary/data-001/test").select("text", "topic")
test_df.cache()
display(test_df)

# COMMAND ----------

exampleResults = deserializedPipeline.transform(test_df)
display(exampleResults)

# COMMAND ----------

# MAGIC %md ### Import to Non-Spark Applications
# MAGIC 
# MAGIC This section demonstrates how to import our model into a non-Spark application.  MLeap provides a [`LeapFrame` API](http://mleap-docs.combust.ml/mleap-runtime/create-leap-frame.html) which is essentially a local `DataFrame`.  After importing our model using MLeap, we can make predictions on data stored in a `LeapFrame` without using a `SparkContext` or `SparkSession`.

# COMMAND ----------

# MAGIC %scala
# MAGIC import ml.combust.bundle.BundleFile
# MAGIC import ml.combust.mleap.runtime.MleapSupport._
# MAGIC import resource._
# MAGIC 
# MAGIC val zipBundleM = (for(bundle <- managed(BundleFile("jar:file:/tmp/mleap_python_model_export/20news_pipeline-json.zip"))) yield {
# MAGIC   bundle.loadMleapBundle().get
# MAGIC }).opt.get

# COMMAND ----------

# MAGIC %scala
# MAGIC val mleapPipeline = zipBundleM.root

# COMMAND ----------

# MAGIC %md Setup a LeapFrame for testing, manually inserting 1 row of data.

# COMMAND ----------

# MAGIC %scala
# MAGIC import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
# MAGIC import ml.combust.mleap.core.types._
# MAGIC 
# MAGIC val schema = StructType(StructField("text", ScalarType.String),
# MAGIC   StructField("topic", ScalarType.String)).get
# MAGIC val data = Seq(Row("From: marshall@csugrad.cs.vt.edu (Kevin Marshall) Subject: Re: Faith and Dogma Organization: Virginia Tech Computer Science Dept, Blacksburg, VA Lines: 96 NNTP-Posting-Host: csugrad.cs.vt.edu   tgk@cs.toronto.edu (Todd Kelley) writes: >In light of what happened in Waco, I need to get something of my >chest. > >Faith and dogma are dangerous. ", "alt.atheism"))
# MAGIC val frame = DefaultLeapFrame(schema, data)

# COMMAND ----------

# MAGIC %md Now you can use the loaded model to make predictions.

# COMMAND ----------

# MAGIC %scala 
# MAGIC val frame2 = mleapPipeline.transform(frame).get
# MAGIC val data2 = frame2.dataset

# COMMAND ----------

# MAGIC %scala
# MAGIC // The prediction is stored in column with index 2:
# MAGIC frame2.schema.fields.zipWithIndex.foreach { case (field, idx) =>
# MAGIC   println(s"$idx $field")
# MAGIC }

# COMMAND ----------

# MAGIC %scala
# MAGIC // Get the prediction for Row 0
# MAGIC data2(0).getDouble(7)

# COMMAND ----------

