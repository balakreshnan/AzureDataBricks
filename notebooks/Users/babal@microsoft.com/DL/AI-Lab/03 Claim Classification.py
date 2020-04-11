# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Claims classification with Azure Databricks
# MAGIC 
# MAGIC In this notebook, you will train a classification model for claim text that will predict `1` if the claim is an auto insurance claim or `0` if it is a home insurance claim. The model will be built using a Deep Neural Network using TensorFlow via the TFLearn library.
# MAGIC 
# MAGIC This notebook will walk you through a simplified text analytic process that consists of:
# MAGIC * Normalizing the training text data
# MAGIC * Extracting the features of the training text as vectors
# MAGIC * Creating and training a DNN based classifier model
# MAGIC * Using the model to predict classifications
# MAGIC 
# MAGIC For reference, to use this notebook in your own Databricks environment, you will need to create libraries, using the [Create Library](https://docs.azuredatabricks.net/user-guide/libraries.html) interface in Azure Databricks, for the following and attach them to your cluster:
# MAGIC 
# MAGIC **TensorFlow**
# MAGIC * Library Source: PyPi
# MAGIC * Package: `tensorflow`
# MAGIC * Select Create
# MAGIC 
# MAGIC **TFLearn**
# MAGIC * Library Source: PyPi
# MAGIC * Package: `tflearn`
# MAGIC * Select Create

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###Prepare modules
# MAGIC 
# MAGIC This notebook will use the TFLearn library to build and train the classifier. In addition, it relies on a supplied helper library that performs common text analytic functions, called textanalytics.

# COMMAND ----------

import numpy as np
import re
import tflearn
from tflearn.data_utils import to_categorical
import nltk
import uuid

# Takes at most a couple of minutes to download all NLTK content
nltk.download("all")

# COMMAND ----------

# MAGIC %md Let's copy locally all the data needed by this notebook. This can take up to a few minutes, depending on the speed of your connection.

# COMMAND ----------

# Create a temporary folder to store locally relevant content for this notebook
tempFolderUUID = uuid.uuid4()
print("UUID is: ", tempFolderUUID)
tempFolderName = '/FileStore/ignite2018_05_03_{0}'.format(tempFolderUUID)
dbutils.fs.mkdirs(tempFolderName)
print('Content files will be saved to {0}'.format(tempFolderName))

import os
filesToDownload = ['claims_labels.txt', 'claims_text.txt', 'contractions.py', 'textanalytics.py']

for fileToDownload in filesToDownload:
  downloadCommand = 'wget -O ''/dbfs{0}/{1}'' ''https://databricksdemostore.blob.core.windows.net/data/05.03/{1}'''.format(tempFolderName, fileToDownload)
  print(downloadCommand)
  os.system(downloadCommand)
  
#List all downloaded files
dbutils.fs.ls(tempFolderName)

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's import some helper code that resides in Python .PY files.
# MAGIC 
# MAGIC Note: This is how you can bring your extra Python code residing on an arbitrary dbfs location into the Databricks notebook.

# COMMAND ----------

import sys
sys.path.append('/dbfs{0}'.format(tempFolderName))

import textanalytics as ta

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Prepare the training data
# MAGIC 
# MAGIC Contoso Ltd has provided a small document containing examples of the text they receive as claim text. They have provided this in a text file with one line per sample claim.
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC Run the following cell to examine the contents of the file. Take a moment to read the claims (you may find some of them rather comical!).

# COMMAND ----------

claims_corpus = [claim for claim in open('/dbfs{0}/claims_text.txt'.format(tempFolderName))]
claims_corpus

# COMMAND ----------

# MAGIC %md
# MAGIC In addition to the claims sample, Contoso Ltd has also provided a document that labels each of the sample claims provided as either 0 ("home insurance claim") or 1 ("auto insurance claim"). This to is presented as a text file with one row per sample, presented in the same order as the claim text. 
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC Run the following cell to examine the contents of the supplied claims_labels.txt file:

# COMMAND ----------

labels = [int(re.sub("\n", "", label)) for label in open('/dbfs{0}/claims_labels.txt'.format(tempFolderName))]
labels

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC As you can see from the above output, the values are integers 0 or 1. In order to use these as labels with which to train our model, we need to convert these integer values to categorical values (think of them like enum's from other programming languages). 
# MAGIC 
# MAGIC We can use the to_categorical method from TFlearn to convert these value into binary categorical values. Run the following cell:

# COMMAND ----------

labels = to_categorical(labels, 2)
labels

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have our claims text and labels loaded, we are ready to begin our first step in the text analytics process, which is to normalize the text. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###Normalize the claims corpus

# COMMAND ----------

# MAGIC %md
# MAGIC The textanalytics module supplied takes care of implementing our desired normalization logic. In summary, what it does is:
# MAGIC * Expand contractions (for example "can't" becomes "cannot")
# MAGIC * Lowercase all text
# MAGIC * Remove special characters (like punctuation)
# MAGIC * Remove stop words (these are words like "a", "an", "the" that add no value)
# MAGIC 
# MAGIC Run the following command and observe how the claim text is modified:

# COMMAND ----------

norm_corpus = ta.normalize_corpus(claims_corpus)
norm_corpus

# COMMAND ----------

# MAGIC %md
# MAGIC ###Feature extraction: vectorize the claims corpus

# COMMAND ----------

# MAGIC %md
# MAGIC Feature extraction in text analytics has the goal of creating a numeric representation of the textual documents. 
# MAGIC 
# MAGIC During feature extraction, a “vocabulary” of unique words is identified and each word becomes a column in the output. In other words, the table is as wide as the vocabulary.
# MAGIC 
# MAGIC Each row represents a document. The value in each cell is typically a measure of the relative importance of that word in the document, where if a word from the vocabular does not appear that cell has a zero value in that column. In other words, the table is as tall as all of the documents in the corpus.
# MAGIC 
# MAGIC This approach enables machine learning algorithms, which operate against arrays of numbers, to also operate against text becasue each text document is now represented as an array of numbers. 
# MAGIC 
# MAGIC Deep learning algorithms operate on tensors, which are also vectors (or arrays of numbers) and so this approach is also valid for preparing text for use with a deep learning algorithm.  
# MAGIC 
# MAGIC Run the following command to see what the vectorized version of the claims in norm_corpus looks like:

# COMMAND ----------

vectorizer, tfidf_matrix = ta.build_feature_matrix(norm_corpus) 
data = tfidf_matrix.toarray()
print(data.shape)
data

# COMMAND ----------

# MAGIC %md
# MAGIC Observe in the above output, that the shape (the dimensions) of the data is 40 rows by 258 columns. You should interpret this as our vectorizer determined that there 258 words in the vocabulary learned from all documents in the set. There are 40 documents in our training set, hence the vectorized output has 40 rows (one array of numbers for each document).

# COMMAND ----------

# MAGIC %md
# MAGIC ###Build the neural network

# COMMAND ----------

# MAGIC %md
# MAGIC Now that you have normalized and extracted the features from training text data, you are ready to build the classifier. In this case, we will build a simple neural network. The network will be 3 layers deep and each node in one layer is connected to every other node in a subsequent layer. This is what is meant by fully connected. We will train the model by applying a regression. 
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC Run the following cell to build the structure for your neural network:

# COMMAND ----------

# With TFLearn should reset the default graph before re-fitting a model, or you may get strange errors. The following two lines are only needed if you end up rebuilding the network or retraining the model, such as might happen if you re-run cells.
import tensorflow as tf
tf.reset_default_graph()

# Build a neural network. 3 layers deep
net = tflearn.input_data(shape=[None, 258])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# COMMAND ----------

# MAGIC %md
# MAGIC Observe in the above, that we declared in the first line that the input data will be 258 columns wide (lining up to our "vocabulary") with an unspecified number of documents tall. This is what is defined by the shape=[None,258]. 
# MAGIC 
# MAGIC Also, take a look at the second to last line which defines the outputs. This is a fully_connected layer as well, but it only has 2 nodes. This is because the output of our neural network has only two possible values. 
# MAGIC 
# MAGIC The layers in between the input data and the final fully_connected layer represent our hidden layers. How many layers and how many nodes for each layer you should have is typically something you arrive at empircally and through iteration, measuring the models performance. As a general rule of thumb, most neural networks have the same dimensions for all of their hidden layers. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train the neural network

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the structure of our neural network, we create an instance of the DNN class providing it our network. This will become our model.
# MAGIC 
# MAGIC Run the following cell:

# COMMAND ----------

# create the model using the DNN model wrapper
model = tflearn.DNN(net)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we are ready to let the DNN learn by fitting it against our training data and labels.
# MAGIC 
# MAGIC Run the following cell to fit your model against the data:

# COMMAND ----------

# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the final output for the value "acc". This stands for accuracy. If you think of random chance as having a 50% accuracy, is you model better than random? 
# MAGIC 
# MAGIC It's OK if it's not much better then random at this point- this is only your first model! The typical data science process would continue with many more iterations taking different actions to improve the model accuracy, including:
# MAGIC - Acquiring more labeled documents for training
# MAGIC - Preparing the text with more sophisticated techniques such as lemmatization
# MAGIC - Adjusting the model hyperparameters, such as the number of layers and number of nodes per layer

# COMMAND ----------

# MAGIC %md
# MAGIC ###Test classifying claims

# COMMAND ----------

# MAGIC %md
# MAGIC Now that you have constructed a model, try it out against a set of claims. Recall that we need to normalize and featurize the text using the exact same pipeline we used during training.
# MAGIC 
# MAGIC Run the following cell to prepare our test data:

# COMMAND ----------

test_claim = ['I crashed my car into a pole.', 'The flood ruined my house.', 'I lost control of my car and fell in the river.']
test_claim = ta.normalize_corpus(test_claim)
test_claim = vectorizer.transform(test_claim)

test_claim = test_claim.toarray()
print(test_claim.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC Now use the model to predict the classification:

# COMMAND ----------

pred = model.predict(test_claim)
pred

# COMMAND ----------

# MAGIC %md
# MAGIC The way to read the above output is that there is one array per document. The first element in an array corresponds to the confidence it has a label of 0 and the second corresponds to the confidence in a label of 1.
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC Another way to examine this output is as labels. Run the following to view the prediction in this way:

# COMMAND ----------

pred_label = model.predict_label(test_claim)
pred_label

# COMMAND ----------

# MAGIC %md
# MAGIC Note in the above, for each array representing a document, the labels are presented in sorted order according to their confidence. So the first element in the array represents the label the model is predicting. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###Model exporting and importing

# COMMAND ----------

# MAGIC %md
# MAGIC Now that you have a working model, you need export the trained model to a file so that it can be used downstream by the deployed web service.
# MAGIC 
# MAGIC To export the model, you run the save command and provide a filename. Run the following cell:

# COMMAND ----------

# create the output folder if it does not exist
model_output_root = '{0}/output/'.format(os.getcwd())
if not os.path.exists(model_output_root):
  os.mkdir(model_output_root)

# create the temp folder under output if it does not exist
model_output_path = '{0}{1}'.format(model_output_root, tempFolderUUID)
if not os.path.exists(model_output_root):
  os.mkdir(model_output_root)

# save the model to the driver node
model_output_fileName = '{0}/claim_classifier.tfl'.format(model_output_path)
model.save(model_output_fileName)
print('Model saved to:', model_output_fileName)

# COMMAND ----------

# MAGIC %md Take note of the tempFolderUUID as you will need this value in the next notebook to deploy the model.

# COMMAND ----------

print(tempFolderUUID)

# COMMAND ----------

# MAGIC %md
# MAGIC To test re-loading the model into the same Databricks Notebook instance, you first need to reset the default TensorFlow graph. 
# MAGIC 
# MAGIC Then run the following cell:

# COMMAND ----------

# Within a Databricks Notebook session wher you have been building a network, 
# you have to reset the default graph before loading a model, or you may get strange errors.
import tensorflow as tf
tf.reset_default_graph()

# COMMAND ----------

# MAGIC %md
# MAGIC Before you can load the saved model, you need to re-create the structure of the neural network. Then you can use the load method to read the file from disk.
# MAGIC 
# MAGIC Run the following cell to load the model:

# COMMAND ----------

# Build the neural network and then load its weights from disk
net2 = tflearn.input_data(shape=[None, 258])
net2 = tflearn.fully_connected(net2, 32)
net2 = tflearn.fully_connected(net2, 32)
net2 = tflearn.fully_connected(net2, 2, activation='softmax')
net2 = tflearn.regression(net2)
model2 = tflearn.DNN(net2)
model2.load(model_output_fileName, weights_only=True)

# COMMAND ----------

# MAGIC %md
# MAGIC As before you can use the model to run predictions. 
# MAGIC 
# MAGIC Run the following cells to try the prediction with the re-loaded model:

# COMMAND ----------

# Test that the loaded model works
pred_label = model2.predict_label(test_claim).tolist()
pred_label

# COMMAND ----------

