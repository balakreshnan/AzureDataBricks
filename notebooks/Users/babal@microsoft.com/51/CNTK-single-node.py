# Databricks notebook source
# MAGIC %md #Use CNTK on a single node
# MAGIC This notebook demonstrates the use of the CNTK package to train a feed-forward convolutional network to recognize digits from the MNIST database. 
# MAGIC 
# MAGIC The contents of this notebook is based on CNTK tutorial [103A](https://cntk.ai/pythondocs/CNTK_103A_MNIST_DataLoader.html) and the original CNTK [103B](https://github.com/Microsoft/CNTK/blob/v2.0.beta15.0/Tutorials/CNTK_103B_MNIST_FeedForwardNetwork.ipynb) tutorial, with minor modifications to run on Databricks. Thanks to the developers of CNTK for this tutorial! 

# COMMAND ----------

# MAGIC %md
# MAGIC NOTE: This notebook has been tested to work with CNTK >= 2.0. To run the notebook, first install CNTK via init script. You can then verify your installation via the cell below:

# COMMAND ----------

import cntk
print("CNTK version: %s"%(cntk.__version__))

# COMMAND ----------

# MAGIC %md # Feedforward network for MNIST
# MAGIC This tutorial is targeted to individuals who are new to CNTK and to machine learning. We assume you have completed or are familiar with [CNTK 101](https://cntk.ai/pythondocs/CNTK_101_LogisticRegression.html) and [102](https://cntk.ai/pythondocs/CNTK_102_FeedForward.html). In this tutorial, you will train a feed forward network based simple model to recognize handwritten digits.
# MAGIC This tutorial is divided into two parts:
# MAGIC 
# MAGIC * Part A: Getting familiar with the MNIST database that will be used later in the tutorial
# MAGIC * Part B: Using the feedforward classifier from CNTK 102 to classify digits in MNIST data set

# COMMAND ----------

# MAGIC %md Import relevant modules to be used later

# COMMAND ----------

from __future__ import print_function
import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import struct
import sys

try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve

# COMMAND ----------

# MAGIC %md ## Part A: Data download
# MAGIC We will download the data onto the driver. The MNIST database is a standard handwritten digits that has been widely used for training and testing of machine learning algorithms. It has a training set of 60,000 images and a test set of 10,000 images with each image being 28 x 28 pixels. This set is easy to use visualize and train on any computer.

# COMMAND ----------

# Functions to load MNIST images and unpack into train and test set.
# - loadData reads image data and formats into a 28x28 long array
# - loadLabels reads the corresponding labels data, 1 for each image
# - load packs the downloaded image and labels data into a combined format to be read later by 
#   CNTK text reader 

def loadData(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))

def loadLabels(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, 1))

def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))

# COMMAND ----------

# MAGIC %md ## Download the data
# MAGIC The MNIST data contains a training and test set; the training set has 60000 images while the test set has 10000 images. Let's download the data:

# COMMAND ----------

# URLs for the train image and labels data
url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
num_train_samples = 60000

print("Downloading train data")
train = try_download(url_train_image, url_train_labels, num_train_samples)


url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
num_test_samples = 10000

print("Downloading test data")
test = try_download(url_test_image, url_test_labels, num_test_samples)

# COMMAND ----------

# MAGIC %md ## Visualize the data

# COMMAND ----------

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(train[i, :-1].reshape(28,28), cmap='Greys_r')
    plt.title(str(train[i,-1]),fontsize=20, fontweight="bold", color="black")
    plt.axis('off')
plt.show()
display()

# COMMAND ----------

# MAGIC %md ## Save the images
# MAGIC Save the images in a local directory. While saving the data we flatten the images to a vector (28x28 image pixels becomes an array of length 784 data points) and the labels are encoded using [1-hot](https://en.wikipedia.org/wiki/One-hot) encoding (e.g. a label of 3 given a label space of 10 possible digits becomes 0010000000.

# COMMAND ----------

# Save the data files into a format compatible with CNTK text reader
def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(filename):
        print("Saving", filename )
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)

# COMMAND ----------

# Save the train and test files (prefer our default path for the data)
data_dir = os.path.join("..", "Examples", "Image", "DataSets", "MNIST")
if not os.path.exists(data_dir):
    data_dir = os.path.join("data", "MNIST")

print ('Writing train text file...')
savetxt(os.path.join(data_dir, "Train-28x28_cntk_text.txt"), train)

print ('Writing test text file...')
savetxt(os.path.join(data_dir, "Test-28x28_cntk_text.txt"), test)

print('Done')

# COMMAND ----------

# MAGIC %md ##Part B: Training a feedforward convolutional network

# COMMAND ----------

import cntk as C
from cntk import Trainer, learning_rate_schedule, UnitType, sgd, input_variable
from cntk.io import CTFDeserializer, MinibatchSource, StreamDef, StreamDefs
from cntk.io import INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.initializer import glorot_uniform
from cntk.layers import default_options, Dense

# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        cntk.device.set_default_device(cntk.device.cpu())
    else:
        cntk.device.set_default_device(cntk.device.gpu(0))

# COMMAND ----------

# Ensure we always get the same amount of randomness
np.random.seed(0)

# Define the data dimensions
input_dim = 784
num_output_classes = 10

# COMMAND ----------

# MAGIC %md ## Data reading
# MAGIC 
# MAGIC We define a ``create_reader`` function to read the training and test data using the [CTF deserializer](https://cntk.ai/pythondocs/cntk.io.html?highlight=ctfdeserializer#cntk.io.CTFDeserializer). The labels are 1-hot encoded.

# COMMAND ----------

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    labelStream = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    deserializer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels = labelStream, features = featureStream))
    return C.io.MinibatchSource(deserializer,
       randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

# COMMAND ----------

# MAGIC %md ## Model Creation
# MAGIC Our feed forward network will be relatively simple with 2 hidden layers (``num_hidden_layers``) with each layer having 200 hidden nodes (``hidden_layers_dim``).
# MAGIC 
# MAGIC <img src="http://cntk.ai/jup/feedforward_network.jpg" width=400 height=100>
# MAGIC 
# MAGIC If you are not familiar with the terms "hidden layer" and "number of hidden layers", please refer back to CNTK 102 tutorial.
# MAGIC For this tutorial: The number of green nodes (refer to picture above) in each hidden layer is set to 200 and the number of hidden layers (refer to the number of layers of green nodes) is 2. Fill in the following values:
# MAGIC 
# MAGIC * ``num_hidden_layers``
# MAGIC * ``hidden_layers_dim``
# MAGIC 
# MAGIC Note: In this illustration, we have not shown the bias node (introduced in the logistic regression tutorial). Each hidden layer would have a bias node.

# COMMAND ----------

num_hidden_layers = 2
hidden_layers_dim = 400

# COMMAND ----------

# MAGIC %md Network input and output: 
# MAGIC - **input** variable (a key CNTK concept): 
# MAGIC >An **input** variable is a container in which we fill different observations in this case image pixels during model learning (a.k.a.training) and model evaluation (a.k.a. testing). Thus, the shape of the `input_variable` must match the shape of the data that will be provided.  For example, when data are images each of  height 10 pixels  and width 5 pixels, the input feature dimension will be 50 (representing the total number of image pixels). More on data and their dimensions to appear in separate tutorials.
# MAGIC 
# MAGIC 
# MAGIC **Question** What is the input dimension of your chosen model? This is fundamental to our understanding of variables in a network or model representation in CNTK.

# COMMAND ----------

input = input_variable(input_dim)
label = input_variable(num_output_classes)

# COMMAND ----------

# MAGIC %md ## Feed forward network setup
# MAGIC 
# MAGIC If you are not familiar with the feedforward network, please refer to CNTK 102. In this tutorial we are using the same network. 

# COMMAND ----------

def create_model(features):
    with default_options(init = glorot_uniform(), activation = C.ops.relu):
            h = features
            for _ in range(num_hidden_layers):
                h = Dense(hidden_layers_dim)(h)
            r = Dense(num_output_classes, activation = None)(h)
            return r
        
z = create_model(input)

# COMMAND ----------

# MAGIC %md `z` will be used to represent the output of a network.
# MAGIC 
# MAGIC We introduced sigmoid function in CNTK 102, in this tutorial you should try different activation functions. You may choose to do this right away and take a peek into the performance later in the tutorial or run the preset tutorial and then choose to perform the suggested activity.
# MAGIC 
# MAGIC 
# MAGIC ** Suggested Activity **
# MAGIC - Record the training error you get with `sigmoid` as the activation function
# MAGIC - Now change to `relu` as the activation function and see if you can improve your training error

# COMMAND ----------

# Scale the input to 0-1 range by dividing each pixel by 256.
z = create_model(input/256.0)

# COMMAND ----------

# MAGIC %md ### Learning model parameters
# MAGIC 
# MAGIC Same as the previous tutorial, we use the `softmax` function to map the accumulated evidences or activations to a probability distribution over the classes (Details of the [softmax function][]).
# MAGIC 
# MAGIC [softmax function]: http://cntk.ai/pythondocs/cntk.ops.html#cntk.ops.softmax

# COMMAND ----------

# MAGIC %md ## Training
# MAGIC 
# MAGIC Similar to CNTK 102, we use minimize the cross-entropy between the label and predicted probability by the network. If this terminology sounds strange to you, please refer to the CNTK 102 for a refresher. 

# COMMAND ----------

loss = C.cross_entropy_with_softmax(z, label)

# COMMAND ----------

# MAGIC %md #### Evaluation
# MAGIC 
# MAGIC In order to evaluate the classification, one can compare the output of the network which for each observation emits a vector of evidences (can be converted into probabilities using `softmax` functions) with dimension equal to number of classes.

# COMMAND ----------

label_error = C.classification_error(z, label)

# COMMAND ----------

# MAGIC %md ### Configure training
# MAGIC 
# MAGIC The trainer strives to reduce the `loss` function by different optimization approaches, [Stochastic Gradient Descent][] (`sgd`) being one of the most popular one. Typically, one would start with random initialization of the model parameters. The `sgd` optimizer would calculate the `loss` or error between the predicted label against the corresponding ground-truth label and using [gradient-decent][] generate a new set model parameters in a single iteration. 
# MAGIC 
# MAGIC The aforementioned model parameter update using a single observation at a time is attractive since it does not require the entire data set (all observation) to be loaded in memory and also requires gradient computation over fewer datapoints, thus allowing for training on large data sets. However, the updates generated using a single observation sample at a time can vary wildly between iterations. An intermediate ground is to load a small set of observations and use an average of the `loss` or error from that set to update the model parameters. This subset is called a *minibatch*.
# MAGIC 
# MAGIC With minibatches we often sample observation from the larger training dataset. We repeat the process of model parameters update using different combination of training samples and over a period of time minimize the `loss` (and the error). When the incremental error rates are no longer changing significantly or after a preset number of maximum minibatches to train, we claim that our model is trained.
# MAGIC 
# MAGIC One of the key parameter for optimization is called the `learning_rate`. For now, we can think of it as a scaling factor that modulates how much we change the parameters in any iteration. We will be covering more details in later tutorial. 
# MAGIC With this information, we are ready to create our trainer. 
# MAGIC 
# MAGIC [optimization]: https://en.wikipedia.org/wiki/Category:Convex_optimization
# MAGIC [Stochastic Gradient Descent]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
# MAGIC [gradient-decent]: http://www.statisticsviews.com/details/feature/5722691/Getting-to-the-Bottom-of-Regression-with-Gradient-Descent.html

# COMMAND ----------

# Instantiate the trainer object to drive the model training
learning_rate = 0.2
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
learner = sgd(z.parameters, lr_schedule)
trainer = Trainer(z, (loss, label_error), [learner])

# COMMAND ----------

# MAGIC %md First let us create some helper functions that will be needed to visualize different functions associated with training.

# COMMAND ----------

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0:4d}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
        
    return mb, training_loss, eval_error

# COMMAND ----------

# MAGIC %md <a id='#Run the trainer'></a>
# MAGIC ### Run the trainer
# MAGIC 
# MAGIC We are now ready to train our fully connected neural net. We want to decide what data we need to feed into the training engine.
# MAGIC 
# MAGIC In this example, each iteration of the optimizer will work on `minibatch_size` sized samples. We would like to train on all 60000 observations. Additionally we will make multiple passes through the data specified by the variable `num_sweeps_to_train_with`. With these parameters we can proceed with training our simple feed forward network.

# COMMAND ----------

# Initialize the parameters for the trainer
minibatch_size = 64
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

# COMMAND ----------

# Create the reader to training data set
train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
reader_train = create_reader(train_file, True, input_dim, num_output_classes)

# Map the data streams to the input and labels.
input_map = {
    label  : reader_train.streams.labels,
    input  : reader_train.streams.features
} 

# Run the trainer on and perform model training
training_progress_output_freq = 500

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_minibatches_to_train)):
    
    # Read a mini batch from the training data file
    data = reader_train.next_minibatch(minibatch_size, input_map = input_map)
    
    trainer.train_minibatch(data)
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
    
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)

# COMMAND ----------

# MAGIC %md Let us plot the errors over the different training minibatches. Note that as we iterate the training loss decreases though we do see some intermediate bumps. 
# MAGIC 
# MAGIC Hence, we use smaller minibatches and using `sgd` enables us to have a great scalability while being performant for large data sets. There are advanced variants of the optimizer unique to CNTK that enable harnessing computational efficiency for real world data sets and will be introduced in advanced tutorials. 

# COMMAND ----------

# Compute the moving average loss to smooth out the noise in SGD
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

# Plot the training loss and the training error
plt.figure()
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--', linewidth=2)
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')
plt.grid("on")
plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--', linewidth=2)
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.grid("on")
plt.tight_layout();
plt.show()
display()

# COMMAND ----------

# MAGIC %md ## Evaluation / Testing 
# MAGIC 
# MAGIC Now that we have trained the network, let us evaluate the trained network on the test data. This is done using `trainer.test_minibatch`.

# COMMAND ----------

# Read the training data
test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
reader_test = create_reader(test_file, False, input_dim, num_output_classes)

test_input_map = {
    label  : reader_test.streams.labels,
    input  : reader_test.streams.features,
}

# Test data for trained model
test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples // test_minibatch_size
test_result = 0.0

for i in range(num_minibatches_to_test):
    
    # We are loading test data in batches specified by test_minibatch_size
    # Each data point in the minibatch is a MNIST digit image of 784 dimensions 
    # with one pixel per dimension that we will encode / decode with the 
    # trained model.
    data = reader_test.next_minibatch(test_minibatch_size,
                                      input_map = test_input_map)

    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))

# COMMAND ----------

# MAGIC %md Note, this error is very comparable to our training error indicating that our model has good "out of sample" error a.k.a. generalization error. This implies that our model can very effectively deal with previously unseen observations (during the training process). This is key to avoid the phenomenon of overfitting.

# COMMAND ----------

# MAGIC %md We have so far been dealing with aggregate measures of error. Let us now get the probabilities associated with individual data points. For each observation, the `eval` function returns the probability distribution across all the classes. The classifier is trained to recognize digits, hence has 10 classes. First let us route the network output through a `softmax` function. This maps the aggregated activations across the network to probabilities across the 10 classes.

# COMMAND ----------

out = C.softmax(z)

# COMMAND ----------

# MAGIC %md Let us predict on a small minibatch sample from the test data.

# COMMAND ----------

# Read the data for evaluation
reader_eval = create_reader(test_file, False, input_dim, num_output_classes)

eval_minibatch_size = 25
eval_input_map = { input  : reader_eval.streams.features } 

data = reader_test.next_minibatch(eval_minibatch_size, input_map = test_input_map)

img_label = data[label].asarray()
img_data = data[input].asarray()
predicted_label_prob = [out.eval(img_data[i,:,:]) for i in range(img_data.shape[0])]

# COMMAND ----------

# Find the index with the maximum value for both predicted as well as the ground truth
pred = [np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob))]
gtlabel = [np.argmax(img_label[i,:,:]) for i in range(img_label.shape[0])]

# COMMAND ----------

print("Label    :", gtlabel[:25])
print("Predicted:", pred)

# COMMAND ----------

# MAGIC %md We can now visualize some of the results:

# COMMAND ----------

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(img_data[i, :, :].reshape(28,28), cmap='Greys_r')
    plt.title(str(pred[i]), fontsize=20, fontweight="bold", color="green" if pred[i] == gtlabel[i] else "red")
    plt.axis("off")
plt.show()
display()