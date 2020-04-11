# Databricks notebook source
# MAGIC %md # Deep Learning Pipelines for Apache Spark - Release 1.4.0
# MAGIC 
# MAGIC Deep Learning Pipelines is a library published by Databricks to provide high-level APIs for scalable deep learning model application and transfer learning via integration of popular deep learning libraries with MLlib Pipelines and Spark SQL. For an overview and the philosophy behind the library, check out the Databricks [blog post](https://databricks.com/blog/2017/06/06/databricks-vision-simplify-large-scale-deep-learning.html). This notebook parallels the [Deep Learning Pipelines README](https://github.com/databricks/spark-deep-learning), detailing usage examples with additional tips for getting started with the library on Databricks.

# COMMAND ----------

# MAGIC %md ## Cluster set-up
# MAGIC 
# MAGIC * Create a Python 3 cluster running Databricks Runtime 5.0 or above.
# MAGIC * Create a `spark-deep-learning` library with the Source option **Maven** and Coordinate `1.4.0-spark2.4-s_2.11`. 
# MAGIC * Create libraries with the Source option **PyPI** and Package `tensorflow==1.12.0`,`keras==2.2.4`, `h5py==2.7.0`, `wrapt`.
# MAGIC * Install the libraries on the cluster. 
# MAGIC 
# MAGIC **Note:** If you are using Databricks Runtime 5.0 ML or above you can skip creating and installing the libraries.
# MAGIC 
# MAGIC Refer to the project's [GitHub page](https://github.com/databricks/spark-deep-learning) for the latest examples and docs.

# COMMAND ----------

# MAGIC %md ## In this notebook
# MAGIC 
# MAGIC Deep Learning Pipelines provides a suite of tools around deep learning. The tools can be categorized as
# MAGIC 
# MAGIC Working with image data:
# MAGIC * **Loading images** natively in Spark DataFrames
# MAGIC * **Transfer learning**, a super quick way to leverage deep learning
# MAGIC * **Distributed hyperparameter tuning** via Spark MLlib Pipelines
# MAGIC * **Applying deep learning models at scale** to images, using your own or known popular models, to make predictions or transform them into features
# MAGIC 
# MAGIC Working with general tensors:
# MAGIC * **Applying deep learning models at scale** to tensors of up to 2 dimensions
# MAGIC 
# MAGIC Deploying Models in SQL:
# MAGIC * **Deploying models as SQL functions** to empower everyone by making deep learning available in SQL
# MAGIC 
# MAGIC We'll cover each one with examples below.

# COMMAND ----------

# MAGIC %md ## Working with image data

# COMMAND ----------

# MAGIC %md Let us first get some images to work with in this notebook. We'll use the flowers dataset from the [TensorFlow retraining tutorial](https://www.tensorflow.org/tutorials/image_retraining).

# COMMAND ----------

# MAGIC %sh 
# MAGIC curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
# MAGIC tar xzf flower_photos.tgz &>/dev/null

# COMMAND ----------

display(dbutils.fs.ls('file:/databricks/driver/flower_photos'))

# COMMAND ----------

# MAGIC %md The `file:/...` directory will be cleared out upon cluster termination. That doesn't matter for this example notebook, but in most cases we'd want to store the images in a more permanent place. 
# MAGIC 
# MAGIC Move the files to DBFS to see how to work with it in the use cases below.

# COMMAND ----------

img_dir = '/tmp/flower_photos'

# COMMAND ----------

dbutils.fs.mkdirs(img_dir)

dbutils.fs.cp('file:/databricks/driver/flower_photos/tulips', img_dir + "/tulips", recurse=True)
dbutils.fs.cp('file:/databricks/driver/flower_photos/daisy', img_dir + "/daisy", recurse=True)
dbutils.fs.cp('file:/databricks/driver/flower_photos/LICENSE.txt', img_dir)

display(dbutils.fs.ls(img_dir))

# COMMAND ----------

# MAGIC %md Create a small sample set of images for quick demonstration.

# COMMAND ----------

sample_img_dir = img_dir + "/sample"
dbutils.fs.rm(sample_img_dir, recurse=True)
dbutils.fs.mkdirs(sample_img_dir)
files =  dbutils.fs.ls(img_dir + "/daisy")[0:10] + dbutils.fs.ls(img_dir + "/tulips")[0:1]
for f in files:
  dbutils.fs.cp(f.path, sample_img_dir)
display(dbutils.fs.ls(sample_img_dir))

# COMMAND ----------

# MAGIC %md ### Loading images
# MAGIC 
# MAGIC The first step to apply deep learning on images is the ability to load the images. Spark and Deep Learning Pipelines include utility functions that can load millions of images into a Spark DataFrame and decode them automatically in a distributed fashion, allowing manipulation at scale.

# COMMAND ----------

# MAGIC %md Using Spark's image data source.

# COMMAND ----------

image_df = spark.read.format("image").load(sample_img_dir)


# COMMAND ----------

display(image_df)

# COMMAND ----------

# MAGIC %md Using a custom image library.

# COMMAND ----------

from sparkdl.image import imageIO
image_df = imageIO.readImagesWithCustomFn(sample_img_dir, decode_f=imageIO.PIL_decode)

# COMMAND ----------

# MAGIC %md The resulting DataFrame contains a string column named "image" containing an image struct with schema == [ImageSchema](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/image/ImageSchema.html).

# COMMAND ----------

display(image_df)

# COMMAND ----------

# MAGIC %md ### Transfer learning
# MAGIC Deep Learning Pipelines provides utilities to perform [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) on images, which is one of the fastest (code and run-time -wise) ways to start using deep learning. Using Deep Learning Pipelines, it can be done in just several lines of code.

# COMMAND ----------

# MAGIC %md First, create training and test DataFrames for transfer learning - this piece of code is longer than transfer learning itself below!

# COMMAND ----------

from pyspark.sql.functions import lit
from sparkdl.image import imageIO

tulips_df = spark.read.format("image").load(img_dir + "/tulips").withColumn("label", lit(1))
daisy_df = imageIO.readImagesWithCustomFn(img_dir + "/daisy", decode_f=imageIO.PIL_decode).withColumn("label", lit(0))
tulips_train, tulips_test, _ = tulips_df.randomSplit([0.005, 0.005, 0.99])  # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
daisy_train, daisy_test, _ = daisy_df.randomSplit([0.005, 0.005, 0.99])     # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
train_df = tulips_train.unionAll(daisy_train)
test_df = tulips_test.unionAll(daisy_test)

# Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
# This ensure that each of the paritions has a small size.
train_df = train_df.repartition(100)
test_df = test_df.repartition(100)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer 

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])

p_model = p.fit(train_df)

# COMMAND ----------

# MAGIC %md **Note:** the training step may take a while on Community Edition - try making a smaller training set in that case.

# COMMAND ----------

# MAGIC %md Let's see how well the model does:

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

tested_df = p_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(tested_df.select("prediction", "label"))))

# COMMAND ----------

# MAGIC %md Not bad for a first try with zero tuning! Furthermore, we can look at where we are making mistakes:

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import expr
def _p1(v):
  return float(v.array[1])
p1 = udf(_p1, DoubleType())

df = tested_df.withColumn("p_1", p1(tested_df.probability))
wrong_df = df.orderBy(expr("abs(p_1 - label)"), ascending=False)
display(wrong_df.select("image.origin", "p_1", "label").limit(10))

# COMMAND ----------

# MAGIC %md ### Distributed Hyperparameter Tuning
# MAGIC Getting the best results in deep learning requires experimenting with different values for training parameters, an important step called hyperparameter tuning. Since Deep Learning Pipelines enables exposing deep learning training as a step in Spark’s machine learning pipelines, users can rely on the hyperparameter tuning infrastructure already built into Spark MLlib.

# COMMAND ----------

# MAGIC %md ##### For Keras users
# MAGIC To perform hyperparameter tuning with a Keras Model, `KerasImageFileEstimator` can be used to build an Estimator and use MLlib’s tooling for tuning the hyperparameters (e.g. CrossValidator). `KerasImageFileEstimator` works with image URI columns (not `ImageSchema` columns) in order to allow for custom image loading and processing functions often used with keras.

# COMMAND ----------

# MAGIC %md Prepare training data: We need the image URIs and the labels.

# COMMAND ----------

from pyspark.sql.functions import lit
from sparkdl.image import imageIO
import pyspark.ml.linalg as spla
import pyspark.sql.types as sptyp
import numpy as np

def CreateTrainImageUriandLabels(image_uris, label, cardinality):
  # Create image categorical labels (integer IDs)
  local_rows = []
  for uri in image_uris:
    label_inds = np.zeros(cardinality)
    label_inds[label] = 1.0
    one_hot_vec = spla.Vectors.dense(label_inds.tolist())
    _row_struct = {"uri": uri, "one_hot_label": one_hot_vec, "label": float(label)}
    row = sptyp.Row(**_row_struct)
    local_rows.append(row)

  image_uri_df = sqlContext.createDataFrame(local_rows)
  return image_uri_df

  
label_cardinality = 2

tulips_files = ["/dbfs" + str(f.path)[5:] for f in dbutils.fs.ls(img_dir + "/tulips")]  # make "local" file paths for images
tulips_uri_df = CreateTrainImageUriandLabels(tulips_files,1,label_cardinality)
daisy_files = ["/dbfs" + str(f.path)[5:] for f in dbutils.fs.ls(img_dir + "/daisy")]  # make "local" file paths for images
daisy_uri_df = CreateTrainImageUriandLabels(daisy_files,0,label_cardinality)

tulips_train, tulips_test, _ = tulips_uri_df.randomSplit([0.005, 0.005, 0.99])  # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
daisy_train, daisy_test, _ = daisy_uri_df.randomSplit([0.005, 0.005, 0.99])     # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
train_df = tulips_train.unionAll(daisy_train)
test_df = tulips_test.unionAll(daisy_test)

# Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
# This ensure that each of the paritions has a small size.
train_df = train_df.repartition(100)
test_df = test_df.repartition(100)

# COMMAND ----------

# MAGIC %md To build the estimator with `KerasImageFileEstimator`, we first need to have a Keras model stored as a file. We can just save the Keras built-in model like InceptionV3.

# COMMAND ----------

from keras.applications import InceptionV3

model = InceptionV3(weights="imagenet")
model.save('/tmp/model-full.h5')  # saves to the local filesystem
# move to a permanent place for future use
dbfs_model_full_path = 'dbfs:/models/model-full.h5'
dbutils.fs.cp('file:/tmp/model-full.h5', dbfs_model_full_path) 

# COMMAND ----------

# MAGIC %md 
# MAGIC Or build up some model:

# COMMAND ----------

from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Flatten(input_shape=(299, 299, 3)))
model.add(Dense(2))
model.add(Activation("softmax"))
model.save('/tmp/model-small.h5')  # saves to the local filesystem
dbfs_model_small_path = 'dbfs:/models/model-small.h5'
dbutils.fs.cp('file:/tmp/model-small.h5', dbfs_model_small_path) 

# COMMAND ----------

# MAGIC %md Then create an image loading function that reads the image data from a URI, preprocesses them, and returns the numerical tensor in the Keras Model input format.

# COMMAND ----------

import PIL.Image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

def load_image_from_uri(local_uri):
  img = (PIL.Image.open(local_uri).convert('RGB').resize((299, 299), PIL.Image.ANTIALIAS))
  img_arr = np.array(img).astype(np.float32)
  img_tnsr = preprocess_input(img_arr[np.newaxis, :])
  return img_tnsr

# COMMAND ----------

# MAGIC %md Now, create a `KerasImageFileEstimator` that takes the saved model file.

# COMMAND ----------

from sparkdl.estimators.keras_image_file_estimator import KerasImageFileEstimator

dbutils.fs.cp(dbfs_model_small_path, 'file:/tmp/model-small-tmp.h5')
estimator = KerasImageFileEstimator(inputCol="uri",
                                    outputCol="prediction",
                                    labelCol="one_hot_label",
                                    imageLoader=load_image_from_uri,
                                    kerasOptimizer='adam',
                                    kerasLoss='categorical_crossentropy',
                                    modelFile='/tmp/model-small-tmp.h5')

# COMMAND ----------

# MAGIC %md Use the model for hyperparameter tuning by doing a grid search using `CrossValidataor`.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = (
  ParamGridBuilder()
  .addGrid(estimator.kerasFitParams, [{"batch_size": 16, "verbose": 0},
                                      {"batch_size": 16, "verbose": 0}])
  .build()
)
mc = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label" )
cv = CrossValidator(estimator=estimator, estimatorParamMaps=paramGrid, evaluator=mc, numFolds=2)

cvModel = cv.fit(train_df)

# COMMAND ----------

# MAGIC %md Now see how well it works on the test data.

# COMMAND ----------

mc.evaluate(cvModel.transform(test_df))

# COMMAND ----------

# MAGIC %md ### Applying deep learning models at scale
# MAGIC 
# MAGIC Spark DataFrames are a natural construct for applying deep learning models to a large-scale dataset. Deep Learning Pipelines provides a set of Spark MLlib Transformers for applying TensorFlow Graphs and TensorFlow-backed Keras Models at scale. The Transformers, backed by the TensorFrames library, efficiently handle the distribution of models and data to Spark workers.

# COMMAND ----------

# MAGIC %md #### Applying deep learning models at scale - to images
# MAGIC Deep Learning Pipelines provides several ways to apply models to images at scale: 
# MAGIC * Popular images models can be applied out of the box, without requiring any TensorFlow or Keras code
# MAGIC * TensorFlow graphs that work on images
# MAGIC * Keras models that work on images

# COMMAND ----------

# MAGIC %md ##### Applying popular image models
# MAGIC There are many well-known deep learning models for images. If the task at hand is very similar to what the models provide (e.g. object recognition with ImageNet classes), or for pure exploration, you can use the Transformer `DeepImagePredictor` by specifying the model name.

# COMMAND ----------

from sparkdl import DeepImagePredictor

image_df = spark.read.format("image").load(sample_img_dir)

predictor = DeepImagePredictor(inputCol="image", outputCol="predicted_labels", modelName="InceptionV3", decodePredictions=True, topK=10)
predictions_df = predictor.transform(image_df)

display(predictions_df.select("predicted_labels", "image.origin"))

# COMMAND ----------

# MAGIC %md Notice that the `predicted_labels` column shows "daisy" as a high probability class for all sample flowers using this base model. However, as can be seen from the differences in the probability values, the neural network has the information to discern the two flower types. Hence the transfer learning example above was able to properly learn the differences between daisies and tulips starting from the base model.

# COMMAND ----------

df = p_model.transform(image_df)
display(df.select("image.origin", (1-p1(df.probability)).alias("p_daisy")))

# COMMAND ----------

# MAGIC %md ##### For TensorFlow users
# MAGIC Deep Learning Pipelines provides an MLlib Transformer that will apply the given TensorFlow Graph to a DataFrame containing a column of images (e.g. loaded using the utilities described in the previous section). Here is a very simple example of how a TensorFlow Graph can be used with the Transformer. In practice, the TensorFlow Graph will likely be restored from files before calling `TFImageTransformer`.

# COMMAND ----------

from pyspark.ml.image import ImageSchema
from sparkdl import TFImageTransformer
import sparkdl.graph.utils as tfx
from sparkdl.transformers import utils
import tensorflow as tf

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    image_arr = utils.imageInputPlaceholder()
    resized_images = tf.image.resize_images(image_arr, (299, 299))
    # the following step is not necessary for this graph, but can be for graphs with variables, etc
    frozen_graph = tfx.strip_and_freeze_until([resized_images], graph, sess, return_graph=True)

transformer = TFImageTransformer(channelOrder='BGR', inputCol="image", outputCol="predictions", 
                                 graph=frozen_graph, inputTensor=image_arr,
                                 outputTensor=resized_images, outputMode="image")

image_df = spark.read.format("image").load(sample_img_dir)
processed_image_df = transformer.transform(image_df)

# COMMAND ----------

# MAGIC %md ##### For Keras users
# MAGIC For applying Keras models in a distributed manner using Spark, [`KerasImageFileTransformer`](link_here) works on TensorFlow-backed Keras models. It 
# MAGIC * Internally creates a DataFrame containing a column of images by applying the user-specified image loading and processing function to the input DataFrame containing a column of image URIs
# MAGIC * Loads a Keras model from the given model file path 
# MAGIC * Applies the model to the image DataFrame
# MAGIC 
# MAGIC The difference in the API from `TFImageTransformer` above stems from the fact that usual Keras workflows have very specific ways to load and resize images that are not part of the TensorFlow Graph.

# COMMAND ----------

# MAGIC %md To use the transformer, you also need to have a Keras model too. Let's use the InceptionV3 model built in previous section.
# MAGIC 
# MAGIC Now on the prediction side:

# COMMAND ----------

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from pyspark.sql.types import StringType
from sparkdl import KerasImageFileTransformer

def loadAndPreprocessKerasInceptionV3(uri):
  # this is a typical way to load and prep images in keras
  image = img_to_array(load_img(uri, target_size=(299, 299)))  # image dimensions for InceptionV3
  image = np.expand_dims(image, axis=0)
  return preprocess_input(image)

dbutils.fs.cp(dbfs_model_full_path, 'file:/tmp/model-full-tmp.h5')
transformer = KerasImageFileTransformer(inputCol="uri", outputCol="predictions",
                                        modelFile='/tmp/model-full-tmp.h5',  # local file path for model
                                        imageLoader=loadAndPreprocessKerasInceptionV3,
                                        outputMode="vector")

files = ["/dbfs" + str(f.path)[5:] for f in dbutils.fs.ls(sample_img_dir)]  # make "local" file paths for images
uri_df = sqlContext.createDataFrame(files, StringType()).toDF("uri")

keras_pred_df = transformer.transform(uri_df)

# COMMAND ----------

display(uri_df)

# COMMAND ----------

display(keras_pred_df.select("uri", "predictions"))

# COMMAND ----------

# MAGIC %md ##Working with general tensors

# COMMAND ----------

# MAGIC %md #### Applying deep learning models at scale - to tensors
# MAGIC Deep Learning Pipelines also provides ways to apply models with tensor inputs (up to 2 dimensions), written in popular deep learning libraries:
# MAGIC * TensorFlow graphs
# MAGIC * Keras models

# COMMAND ----------

# MAGIC %md ##### For TensorFlow users
# MAGIC 
# MAGIC `TFTransformer` applies a user-specified TensorFlow graph to tensor inputs of up to 2 dimensions.
# MAGIC The TensorFlow graph may be specified as TensorFlow graph objects (`tf.Graph` or `tf.GraphDef`) or checkpoint or `SavedModel` objects
# MAGIC (see the [input object class](https://github.com/databricks/spark-deep-learning/blob/master/python/sparkdl/graph/input.py#L27) for more detail).
# MAGIC The `transform()` function applies the TensorFlow graph to a column of arrays (where an array corresponds to a Tensor) in the input DataFrame
# MAGIC and outputs a column of arrays corresponding to the output of the graph.

# COMMAND ----------

# MAGIC %md First generate sample dataset of 2-dimensional points, Gaussian distributed around two different centers.

# COMMAND ----------

import numpy as np
from pyspark.sql.types import Row

n_sample = 1000
center_0 = [-1.5, 1.5]
center_1 = [1.5, -1.5]

def to_row(args):
  xy, l = args
  return Row(inputCol = xy, label = l)

samples_0 = [np.random.randn(2) + center_0 for _ in range(n_sample//2)]
labels_0 = [0 for _ in range(n_sample//2)]
samples_1 = [np.random.randn(2) + center_1 for _ in range(n_sample//2)]
labels_1 = [1 for _ in range(n_sample//2)]

# Make dataframe for the Spark use case
rows = map(to_row, zip(map(lambda x: x.tolist(), samples_0 + samples_1), labels_0 + labels_1))
sdf = spark.createDataFrame(rows)

# COMMAND ----------

# MAGIC %md Next, write a function that returns a TensorFlow graph and its input.

# COMMAND ----------

import tensorflow as tf

def build_graph(sess, w0):
  X = tf.placeholder(tf.float64, shape=[None, 2], name="input_tensor")
  model = tf.sigmoid(tf.matmul(X, w0), name="output_tensor")
  return model, X


# COMMAND ----------

# MAGIC %md Following is the code you would write to predict using tensorflow on a single node.

# COMMAND ----------

w0 = np.array([[1], [-1]]).astype(np.float64)
with tf.Session() as sess:
  model, X = build_graph(sess, w0)
  output = sess.run(model, feed_dict = {
    X : samples_0 + samples_1
  })

# COMMAND ----------

# MAGIC %md Now you can use the following Spark MLlib Transformer to apply the model to a DataFrame in a distributed fashion.

# COMMAND ----------

from sparkdl import TFTransformer
from sparkdl.graph.input import TFInputGraph
import sparkdl.graph.utils as tfx

graph = tf.Graph()
with tf.Session(graph=graph) as session, graph.as_default():
    _, _ = build_graph(session, w0)
    gin = TFInputGraph.fromGraph(session.graph, session,
                                 ["input_tensor"], ["output_tensor"])

transformer = TFTransformer(
    tfInputGraph=gin,
    inputMapping={'inputCol': tfx.tensor_name("input_tensor")},
    outputMapping={tfx.tensor_name("output_tensor"): 'outputCol'})

odf = transformer.transform(sdf)

# COMMAND ----------

odf.limit(5).show(truncate = False)

# COMMAND ----------

# MAGIC %md ##### For Keras users
# MAGIC `KerasTransformer` applies a TensorFlow-backed Keras model to tensor inputs of up to 2 dimensions. It loads a Keras model from a given model file path and applies the model to a column of arrays (where an array corresponds to a Tensor), outputting a column of arrays.

# COMMAND ----------

from sparkdl import KerasTransformer
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from pyspark.sql.types import *

# Generate random input data
num_features = 10
num_examples = 100
input_data = [{"features" : np.random.randn(num_features).astype(float).tolist()} for i in range(num_examples)]
schema = StructType([ StructField("features", ArrayType(FloatType()), True)])
input_df = sqlContext.createDataFrame(input_data, schema)

# Create and save a single-hidden-layer Keras model for binary classification
# NOTE: In a typical workflow, we'd train the model before exporting it to disk,
# but we skip that step here for brevity
model = Sequential()
model.add(Dense(units=20, input_shape=[num_features], activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model_path = "/tmp/simple-binary-classification"
model.save(model_path)

# Create transformer and apply it to our input data
transformer = KerasTransformer(inputCol="features", outputCol="predictions", modelFile=model_path)
final_df = transformer.transform(input_df)

# COMMAND ----------

display(final_df)

# COMMAND ----------

# MAGIC %md ## Deploying Models in SQL

# COMMAND ----------

# MAGIC %md ### Deploying models as SQL functions
# MAGIC 
# MAGIC One way to productionize a model is to deploy it as a Spark SQL user defined function (UDF), which allows anyone who knows SQL to use it. Deep Learning Pipelines provides mechanisms to take a deep learning model and *register* a Spark SQL UDF. In particular, Deep Learning Pipelines supports creating SQL UDFs from Keras models that work on image data. The resulting UDF takes a column (formatted as an image struct `ImageSchema`) and produces the output of the given Keras model; e.g. for Inception V3, it produces a real valued score vector over the ImageNet object categories.

# COMMAND ----------

# MAGIC %md You can register a UDF for a Keras model that works on images as follows:

# COMMAND ----------

from keras.applications import InceptionV3
from sparkdl.udf.keras_image_model import registerKerasImageUDF

registerKerasImageUDF("inceptionV3_udf", InceptionV3(weights="imagenet"))

# COMMAND ----------

# MAGIC %md Alternatively, you can also register a UDF from a model file:

# COMMAND ----------

registerKerasImageUDF("my_custom_keras_model_udf", "/tmp/model-full-tmp.h5")

# COMMAND ----------

# MAGIC %md In Keras workflows dealing with images, it's common to have preprocessing steps before the model is applied to the image. If your workflow requires preprocessing, you can optionally provide a preprocessing function to UDF registration. The preprocessor should take in a filepath and return an image array; below is a simple example.

# COMMAND ----------

from keras.applications import InceptionV3
from sparkdl.udf.keras_image_model import registerKerasImageUDF

def keras_load_img(fpath):
    from keras.preprocessing.image import load_img, img_to_array
    import numpy as np
    img = load_img(fpath, target_size=(299, 299))
    return img_to_array(img).astype(np.uint8)

registerKerasImageUDF("inceptionV3_udf_with_preprocessing", InceptionV3(weights="imagenet"), keras_load_img)

# COMMAND ----------

# MAGIC %md Once you register a UDF, you can use it in a SQL query.

# COMMAND ----------

image_df = spark.read.format("image").load(sample_img_dir)
image_df.registerTempTable("sample_images")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT inceptionV3_udf(image) as predictions from sample_images

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT my_custom_keras_model_udf(image) as predictions from sample_images

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT inceptionV3_udf_with_preprocessing(image) as predictions from sample_images

# COMMAND ----------

# MAGIC %md ### Clean up data generated for this notebook

# COMMAND ----------

dbutils.fs.rm(img_dir, recurse=True)
dbutils.fs.rm(dbfs_model_full_path)
dbutils.fs.rm(dbfs_model_small_path)

# COMMAND ----------

# MAGIC %md ### Resources
# MAGIC * See the Databricks [blog post](https://databricks.com/blog/2017/06/06/databricks-vision-simplify-large-scale-deep-learning.html) announcing Deep Learning Pipelines for a high-level overview and more in-depth discussion of some of the concepts here.
# MAGIC * Check out the [Deep Learning Pipelines GitHub page](https://github.com/databricks/spark-deep-learning).
# MAGIC * Learn more about [deep learning on Databricks](https://docs.databricks.com/applications/deep-learning/index.html).