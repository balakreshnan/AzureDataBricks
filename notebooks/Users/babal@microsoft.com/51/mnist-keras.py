# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed deep learning training using Keras with `HorovodRunner` for MNIST
# MAGIC 
# MAGIC This notebook demonstrates how to migrate a single node deep learning (DL) code with Keras to distributed training with Horovod to Databricks with `HorovodRunner`.
# MAGIC 
# MAGIC This guide consists of the following sections:
# MAGIC 
# MAGIC * **Setup**
# MAGIC * **Prepare Single Node Code** 
# MAGIC * **Migrate to Horovod**
# MAGIC * **Migrate to HorovodRunner** 
# MAGIC 
# MAGIC 
# MAGIC * The notebook runs without code changes on CPU or GPU-enabled Apache Spark clusters of two or more machines.
# MAGIC * To run the notebook, create a cluster with
# MAGIC   - At least 4 workers. Otherwise, reduce the `np` for distributed DL.
# MAGIC   - Databricks Runtime 5.0 ML or above.

# COMMAND ----------

import numpy as np
import time
import datetime
import json
import os
import shutil
import tensorflow as tf
from tensorflow.contrib import learn

import horovod.keras as hvd
import keras
from keras import backend as K
from keras import layers
from keras.datasets import mnist
from sparkdl import HorovodRunner

print(tf.__version__)
print(keras.__version__)
if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the TensorFlow backend,'
                       ' because it requires the Datset API, which is not'
                       ' supported on other platforms.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Set up some parameters for training.

# COMMAND ----------



# COMMAND ----------

max_iteration = 500
learning_rate = 0.001
save_checkpoint_steps = 100
logging_steps = 100
batch_size = 128
buffer_size = 1000
epochs = 5
num_classes = 10
hidden2 = 32

# COMMAND ----------

# MAGIC %md
# MAGIC Set the dataset location, checkpoint location, and log location to the shared space.
# MAGIC 
# MAGIC **Note:** Keras is not able to write checkpoints through the FUSE mount, so they are written to a temp directory `/tmp/keras-checkpoints.ckpt` and then copied to blob storage `FUSE_MOUNT_LOCATION + '/MNISTDemo/keras-checkpoints.ckpt'`.

# COMMAND ----------

FUSE_MOUNT_LOCATION = '<FUSE mount>'
working_path = FUSE_MOUNT_LOCATION + '/MNISTDemo/'
tfrecords_train_data_path = FUSE_MOUNT_LOCATION + '/MNISTDemo/mnistData/train.tfrecords'
tfrecords_data_path = FUSE_MOUNT_LOCATION + '/MNISTDemo/mnistData/'
tmp_checkpoint_dir = '/tmp/keras-checkpoints.ckpt'
checkpoint_dir = FUSE_MOUNT_LOCATION + '/MNISTDemo/keras-checkpoints.ckpt'
log_dir = FUSE_MOUNT_LOCATION + '/MNISTDemo/'
timeline_path = FUSE_MOUNT_LOCATION + '/MNISTDemo/MNIST_timeline.json'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparing the data

# COMMAND ----------

# MAGIC %md
# MAGIC We recommend using TFRecord files a data source to save training data to blob storage.
# MAGIC 
# MAGIC * **Dataset to TFRecords:** Tensorflow provides `tf.python_io.TFRecordWriter` to convert the data into TFRecords
# MAGIC 
# MAGIC * **Databricks DataFrames to TFRecords:** We can use `spark-tensorflow-connector` to save Databricks DataFrames to TFRecords. 
# MAGIC 
# MAGIC ``` python
# MAGIC df.write.format('tfrecords').option('recordType', 'Example').mode('overwrite').save(path)
# MAGIC ```
# MAGIC 
# MAGIC `spark-tensorflow-connector` is a library within the `TensorFlow ecosystem` that enables conversion between Spark DataFrames and TFRecords. For more details, check their [GitHub page](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector).

# COMMAND ----------

# MAGIC %md 
# MAGIC Once you have the TFRecord files, you can parse it to a TFRecordDataset. The code to convert MNIST data to TFRecords can be found at [tensorflow/examples/how_tos/reading_data/convert_to_records.py](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/examples/how_tos/reading_data/convert_to_records.py).

# COMMAND ----------

def decode(serialized_example):
  """Parses an image and label from the given `serialized_example`."""
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape((28*28))

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)
  return image, label

def normalize(image, label):
  """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  return image, label

# COMMAND ----------

# MAGIC %md
# MAGIC ## <img src=https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Keras_Logo.jpg/180px-Keras_Logo.jpg alt="tensorflow" height="80" width="80"> 
# MAGIC ## Prepare Single Node Code
# MAGIC 
# MAGIC First you need to have working single-node TensorFlow code. This is modified from the [Keras MNIST Example](https://github.com/keras-team/keras/blob/master/examples/mnist_dataset_api.py).

# COMMAND ----------

# MAGIC %md
# MAGIC Define the Keras model.

# COMMAND ----------

def cnn_layers(inputs):
    num_classes = 10
    x = layers.Reshape((28,28,1))(inputs)
    x = layers.Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return predictions

# COMMAND ----------

def run_training():
  steps_per_epoch = int(np.ceil(6000 / float(batch_size)))  # = 469


  # Create the dataset and its associated one-shot iterator.
  dataset = tf.data.TFRecordDataset(tfrecords_train_data_path)
  dataset = dataset.map(decode)
  dataset = dataset.map(normalize)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
    
  # Model creation using tensors from the get_next() graph node.
  inputs, targets = iterator.get_next()
  model_input = layers.Input(tensor=inputs)
  model_output = cnn_layers(model_input)
  targets = tf.one_hot(targets, num_classes)
  train_model = keras.models.Model(inputs=model_input, outputs=model_output)
  optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5)

  train_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      target_tensors=[targets])
  train_model.summary()
  callbacks = [
    # Reduce the learning rate if training plateaues.
    keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
    keras.callbacks.ModelCheckpoint(tmp_checkpoint_dir, save_weights_only=True)
  ]
  train_model.fit(epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks)
  shutil.copyfile(tmp_checkpoint_dir, checkpoint_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC You can test it on the driver.

# COMMAND ----------

run_training()

# COMMAND ----------

dbutils.fs.rm('file:'+checkpoint_dir,recurse=True)
dbutils.fs.rm('file:'+tmp_checkpoint_dir,recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## <img src=https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Keras_Logo.jpg/180px-Keras_Logo.jpg alt="keras" height="120" width="120"><img src=http://www.clker.com/cliparts/9/b/2/8/11949855961746203850arrow-right-blue_benji_p_01.svg.hi.png height="120" width="60"> <img src=https://user-images.githubusercontent.com/16640218/34506318-84d0c06c-efe0-11e7-8831-0425772ed8f2.png alt="horovod logo" height="120" width="120"> 
# MAGIC ## Migrate to Horovod 
# MAGIC 
# MAGIC Below, we show how to modify your single-machine code to use Horovod. For an additional reference, also check out the [Horovod usage](https://github.com/uber/horovod#usage) guide.

# COMMAND ----------

def run_training_horovod():
  steps_per_epoch = int(np.ceil(6000 / float(batch_size)))  # = 469

  ##############################
  # Horovod: initialize Horovod.
  hvd.init()
  ##############################
  
  #########################################################################
  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = str(hvd.local_rank())
  sess = tf.Session(config=config)
  K.set_session(sess)
  #########################################################################

  dataset = tf.data.TFRecordDataset(tfrecords_train_data_path)
  dataset = dataset.map(decode)
  dataset = dataset.map(normalize)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()

  inputs, targets = iterator.get_next()
  model_input = layers.Input(tensor=inputs)
  model_output = cnn_layers(model_input)
  targets = tf.one_hot(targets, num_classes)
  train_model = keras.models.Model(inputs=model_input, outputs=model_output)
  
  ########################################################
  # Horovod: adjust learning rate based on number of GPUs.
  optimizer= keras.optimizers.Adadelta(1.0 * hvd.size()) # keras.optimizers.RMSprop(lr=2e-3 * hvd.size(), decay=1e-5)
  ########################################################

  #############################################
  # Horovod: add Horovod Distributed Optimizer.
  optimizer = hvd.DistributedOptimizer(optimizer)
  #############################################
  sess.run(tf.global_variables_initializer())
  
  train_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      target_tensors=[targets])
  train_model.summary()
  callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    
    # Reduce the learning rate if training plateaues.
    keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
  ]
    
  # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
  if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(tmp_checkpoint_dir, save_weights_only=True))
  
  train_model.fit(epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks)
  if hvd.rank() == 0:
    shutil.copyfile(tmp_checkpoint_dir, checkpoint_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC Now you can test the Horovod training function.

# COMMAND ----------

run_training_horovod()

# COMMAND ----------

dbutils.fs.rm('file:'+checkpoint_dir,recurse=True)
dbutils.fs.rm('file:'+tmp_checkpoint_dir,recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## <img src=http://files.idg.co.kr/itworld/image/2015/03/spark_logo.jpg alt="spark logo" height="240" width="170"><img src=https://vignette.wikia.nocookie.net/i-shall-seal-the-heavens/images/2/2f/Plus.png/revision/latest?cb=20180221183139 alt="plus" height="50" width="50">      <img src=https://user-images.githubusercontent.com/16640218/34506318-84d0c06c-efe0-11e7-8831-0425772ed8f2.png alt="horovod logo" height="120" width="120"> 
# MAGIC ## Migrate to `HorovodRunner`
# MAGIC 
# MAGIC HorovodRunner takes a Python method that contains DL training code w/ Horovod hooks. This method gets pickled on the driver and sent to Spark workers. A Horovod MPI job is embedded as a Spark job using barrier execution mode.

# COMMAND ----------

# MAGIC %md
# MAGIC If you have the function with Horovod, you can easily build the `HorovodRunner` and run distributed.

# COMMAND ----------

help(HorovodRunner)

# COMMAND ----------

# MAGIC %md
# MAGIC Run HorovodRunner in local mode first.

# COMMAND ----------

hr = HorovodRunner(np=-1)
hr.run(run_training_horovod)

# COMMAND ----------

dbutils.fs.rm('file:'+checkpoint_dir,recurse=True)
dbutils.fs.rm('file:'+tmp_checkpoint_dir,recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Now you can go distributed.

# COMMAND ----------

hr = HorovodRunner(np=0)
hr.run(run_training_horovod)

# COMMAND ----------

dbutils.fs.rm('file:'+checkpoint_dir,recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Horovod Timeline
# MAGIC 
# MAGIC HorovodRunner has the ability to record the timeline of its activity with Horovod 
# MAGIC Timeline. To record a Horovod Timeline, set the `HOROVOD_TIMELINE` environment variable 
# MAGIC to the location of the timeline file to be created. You can then open the timeline file 
# MAGIC using the `chrome://tracing` facility of the Chrome browser.
# MAGIC  
# MAGIC  
# MAGIC **Note:** The `timeline_path` needs to be on a FUSE mount of blob storage.

# COMMAND ----------

timeline_path = FUSE_MOUNT_LOCATION + '/MNISTDemo/MNIST_timeline.json'
import os
os.environ['HOROVOD_TIMELINE'] = '/mnt/databricks-mllib/tmp/horovod_timeline.json'
hr.run(run_training_horovod)

# COMMAND ----------

display(dbutils.fs.ls('file:'+timeline_path))

# COMMAND ----------

dbutils.fs.rm('file:'+checkpoint_dir,recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Copy the file from the FUSE mounted location to the DBFS `/FileStore` folder.

# COMMAND ----------

dbutils.fs.cp('/tmp/MNIST_timeline.png', '/FileStore/MNISTDemo/MNIST_timeline.png')

# COMMAND ----------

# MAGIC %md
# MAGIC Files stored in `/FileStore` are accessible in your web browser at the path `https://<databricks-instance-name>/files`.
# MAGIC 
# MAGIC 
# MAGIC 1. The timeline file is accessible at: `https://<databricks-instance-name>/files/MNISTDemo/MNIST_timeline.json`. Open this URL in your browser and the file will download.
# MAGIC 2. In Chrome, open `chrome://tracing` and then click **Load** to load the file `MNIST_timeline.json` you downloaded. You should see a timeline like the following:

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='https://docs.databricks.com/_static/images/distributed-deep-learning/MNIST-timeline.png'>

# COMMAND ----------

# MAGIC %md For a guide to interpreting the timeline, see [Horovod 
# MAGIC Timeline](https://github.com/uber/horovod/blob/master/docs/timeline.md).