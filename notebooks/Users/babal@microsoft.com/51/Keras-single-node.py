# Databricks notebook source
# MAGIC %md ## Use Keras with TensorFlow on a single node
# MAGIC 
# MAGIC This notebook demonstrates how to use Keras (with TensorFlow in the backend) on the Spark driver node to fit a neural network on MNIST handwritten digit recognition data.
# MAGIC 
# MAGIC Prerequisites:
# MAGIC * A GPU-enabled cluster on Databricks.
# MAGIC * Keras and TensorFlow installed with GPU support.
# MAGIC 
# MAGIC The content of this notebook is [copied from the Keras project](https://github.com/fchollet/keras/blob/47350dc6078053403c59e8da3fd63ac3ae12b5ec/examples/mnist_cnn.py) under the [MIT license](https://github.com/fchollet/keras/blob/47350dc6078053403c59e8da3fd63ac3ae12b5ec/LICENSE) with slight modifications in comments. Thanks to the developers of Keras for this example!

# COMMAND ----------

# MAGIC %md ### Handwritten Digit Recognition
# MAGIC 
# MAGIC This tutorial guides you through a classic computer vision application: identify hand written digits with neural networks. 
# MAGIC We will train a simple Convolutional Neural Network on the MNIST dataset.
# MAGIC 
# MAGIC Note that we will not explicitly choose a backend for Keras, so it will use TensorFlow by default.

# COMMAND ----------

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# COMMAND ----------

# MAGIC %md ### Load and process data
# MAGIC 
# MAGIC We first fetch the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, which is a commonly used dataset for handwritten digit recognition. Keras provides a handy function for loading this data.

# COMMAND ----------

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# COMMAND ----------

# input image dimensions
img_rows, img_cols = 28, 28
# number of classes (digits) to predict
num_classes = 10

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# COMMAND ----------

# MAGIC %md ### Train a CNN model

# COMMAND ----------

# MAGIC %md First, define the model structure.

# COMMAND ----------

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# COMMAND ----------

# MAGIC %md Now, we can fit the model.  This should take about 10-15 seconds per epoch on a commodity GPU, or about 2-3 minutes for 12 epochs.

# COMMAND ----------

batch_size = 128
epochs = 12

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# COMMAND ----------

# MAGIC %md ### Evaluate the model
# MAGIC 
# MAGIC We can get test accuracy above `95%` after 12 epochs, but there is still a lot of margin for improvements via parameter tuning.

# COMMAND ----------

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])