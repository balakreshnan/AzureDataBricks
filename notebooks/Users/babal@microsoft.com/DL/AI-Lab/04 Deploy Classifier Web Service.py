# Databricks notebook source
# MAGIC %md #Install the Azure ML SDK on your Azure Databricks Cluster

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The `Azure Machine Learning Python SDK` is required for leveraging the experimentation, model management and model deployment capabilities of Azure Machine Learning services.
# MAGIC 
# MAGIC If your cluster is not already provisioned with the Azure Machine Learning Python SDK, you easily add it to your cluster by adding the following libraries. 
# MAGIC 
# MAGIC For reference, to use this notebook in your own Databricks environment, you will need to create libraries, using the [Create Library](https://docs.azuredatabricks.net/user-guide/libraries.html) interface in Azure Databricks, for the following and attach them to your cluster:
# MAGIC 
# MAGIC **azureml-sdk**
# MAGIC * Source: Upload Python Egg or PyPi
# MAGIC * PyPi Name: `azureml-sdk[databricks]`
# MAGIC * Select Install Library

# COMMAND ----------

# MAGIC %md
# MAGIC Verify that the Azure ML SDK is installed on your cluster by running the following cell:

# COMMAND ----------

import azureml.core
azureml.core.VERSION

# COMMAND ----------

# MAGIC %md
# MAGIC If you see a version number output in the above cell, your cluster is ready to go.

# COMMAND ----------

# MAGIC %md #Deploy model to Azure Container Instance (ACI)

# COMMAND ----------

# MAGIC %md In this notebook, you will deploy the model you created previously as a web service hosted in Azure Container Service.

# COMMAND ----------

# MAGIC %md ## Locate the model on the driver node

# COMMAND ----------

# MAGIC %md You previously saved the model in DBFS, but to deploy it using Azure Machine Learning services, you will need to copy the model to local storage on the driver node.
# MAGIC 
# MAGIC Update `tempFolderUUID` in the following cell with the UUID from the previous notebook. 

# COMMAND ----------

import os
##NOTE: service deployment always gets the model from the current working dir. 
tempFolderUUID = "0aa683e8-79f4-4853-ae2b-a4b673419a17" # UUID from previous notebook
tempFolderName = 'output/{0}'.format(tempFolderUUID)

model_name = "claim_classifier.tfl"
relative_model_path_local = tempFolderName
print("Using relative path to model output folder: ", relative_model_path_local)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the model with Azure Machine Learning

# COMMAND ----------

# MAGIC %md Begin by loading your Azure Machine Learning Workspace configuration from disk.

# COMMAND ----------

import azureml.core
from azureml.core.workspace import Workspace

#get the config file from dbfs
aml_config = '/aml_config'
dbutils.fs.cp(aml_config, 'file:'+os.getcwd()+aml_config, recurse=True)

ws = Workspace.from_config()

# COMMAND ----------

# MAGIC %md In the following, you register the model file with Azure Machine Learning (which saves a copy of the model in the cloud).

# COMMAND ----------

#Register the model
from azureml.core.model import Model
mymodel = Model.register(model_path = relative_model_path_local, # this points to a local file or folder in the current working dir
                       model_name = model_name, # this is the name the model is registered with                 
                       description = "Claims classification model.",
                       workspace = ws)

print(mymodel.name, mymodel.description, mymodel.version)

# COMMAND ----------

# MAGIC %md #Create the scoring web service

# COMMAND ----------

# MAGIC %md When deploying models for scoring with Azure Machine Learning services, you need to define the code for a simple web service that will load your model and use it for scoring. By convention this service has two methods `init` which loads the model and `run` which scores data using the loaded model. 
# MAGIC 
# MAGIC This scoring service code will later be deployed inside of a specially prepared Docker container.

# COMMAND ----------

#%%writefile score_classifer.py
score_classifier = """

import tflearn
import numpy as np
import nltk
import os
import sys
import urllib.request

def init():
    try:
        print("init() called.")
        
        global vectorizer, model2
        
        # Takes at most a couple of minutes to download all NLTK content
        print("downloading nltk.")
        nltk.download("all")
        
        tempFolderName = './resources'
        os.mkdir(tempFolderName)
        print('Content files will be saved to {0}'.format(tempFolderName))

        print("downloading files...")
        filesToDownload = ['claims_labels.txt', 'claims_text.txt', 'contractions.py', 'textanalytics.py']
        for fileToDownload in filesToDownload:
          file_url = "https://databricksdemostore.blob.core.windows.net/data/05.03/{0}".format(fileToDownload)
          dest_path = "{0}/{1}".format(tempFolderName, fileToDownload)
          print("attempting to download: ", file_url, " to ", dest_path)
          urllib.request.urlretrieve(file_url, dest_path)
          print("download succeeded.")
        print("file download complete.")
        
        print("importing textanalytics...")
        sys.path.append(os.path.abspath('./{0}'.format(tempFolderName)))      
        import textanalytics as ta
        print("importing succeeded.")
        
        claims_corpus = [claim for claim in open("{0}/{1}".format(tempFolderName, "claims_text.txt"))]

        norm_corpus = ta.normalize_corpus(claims_corpus)
        vectorizer, tfidf_matrix = ta.build_feature_matrix(norm_corpus) 
        
        print("creating network")
        # Build the neural network and then load its weights from disk
        net2 = tflearn.input_data(shape=[None, 258])
        net2 = tflearn.fully_connected(net2, 32)
        net2 = tflearn.fully_connected(net2, 32)
        net2 = tflearn.fully_connected(net2, 2, activation='softmax')
        net2 = tflearn.regression(net2)
        
        print("loading model from disk...")
        from azureml.core.model import Model
        model2 = tflearn.DNN(net2)
        model_name = "claim_classifier.tfl" 
        model_path = Model.get_model_path(model_name)
        model2.load('{0}/{1}'.format(model_path, model_name), weights_only=True)
        print("model loaded.")
        
        print("init complete.")

    except Exception as e:
        print("Exception in init: " + str(e))
        trainedModel = e

def run(input_claim_str):
    response = '' 
    global vectorizer, model2

    try:
        print("calling run with: " + input_claim_str)     
        
        print("importing textanalytics...")
        import textanalytics as ta
        
        print("preparing text")
        claims = [input_claim_str]
        claims = ta.normalize_corpus(claims)
        claims = vectorizer.transform(claims)
        claims = claims.toarray()
        
        print("calling model2.predict_label")
        pred_label = model2.predict_label(claims)
        
        response = str(pred_label[0][0])
        
        print("response: " + str(response))
        
    except Exception as e:
        print("Exception in run: " + str(e))
        return (str(e))

    # Return results
    return response
    
"""

exec(score_classifier)

with open("score_classifier.py", "w") as file:
    file.write(score_classifier)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a Conda dependencies environment file

# COMMAND ----------

# MAGIC %md Your scoring service can have dependencies install by using a Conda environment file. Items listed in this file will be conda or pip installed within the Docker container that is created and thus be available to your scoring web service logic.

# COMMAND ----------

from azureml.core.conda_dependencies import CondaDependencies 

myacienv = CondaDependencies.create(conda_packages=['scikit-learn','numpy','pandas'], pip_packages=['nltk','tensorflow','tflearn','azureml-sdk'])

with open("mydeployenv.yml","w") as f:
    f.write(myacienv.serialize_to_string())

# COMMAND ----------

# MAGIC %md #Deployment

# COMMAND ----------

# MAGIC %md In the following cells you will use the Azure Machine Learning SDK to package the model and scoring script in a container, and deploy that container to an Azure Container Instance.
# MAGIC 
# MAGIC Run the following cells.

# COMMAND ----------

# MAGIC %md Create a configuration of the ACI web service instance that provides the number of CPU cores, size of memory, a collection of tags and a description.

# COMMAND ----------

from azureml.core.webservice import AciWebservice, Webservice

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 1, 
    memory_gb = 1, 
    tags = {'name':'Claim Classification'}, 
    description = 'Classifies a claim as home or auto.')

# COMMAND ----------

# MAGIC %md Next, build up a container image configuration that names the scoring service script, the runtime (python or Spark), and provides the conda file.

# COMMAND ----------

service_name = "claimclassservice"
runtime = "python" #"python" #
driver_file = "score_classifier.py"
conda_file = "mydeployenv.yml"

from azureml.core.image import ContainerImage

image_config = ContainerImage.image_configuration(execution_script = driver_file,
                                                  runtime = runtime,
                                                  conda_file = conda_file)

# COMMAND ----------

# MAGIC %md Now you are ready to begin your deployment to the Azure Container Instance. 
# MAGIC 
# MAGIC Run the following cell. This may take between **5-15 minutes** to complete.
# MAGIC 
# MAGIC You will see output similar to the following when your web service is ready:
# MAGIC `SucceededACI service creation operation finished, operation "Succeeded"`

# COMMAND ----------

webservice = Webservice.deploy_from_model(
  workspace=ws, 
  name=service_name, 
  deployment_config=aci_config,
  models = [mymodel], 
  image_config=image_config, 
  )

webservice.wait_for_deployment(show_output=True)

# COMMAND ----------

# MAGIC %md #Test the deployed service

# COMMAND ----------

# MAGIC %md Now you are ready to test scoring using the deployed web service. The following cell invokes the web service. 
# MAGIC 
# MAGIC Run the following cells to test scoring using a single input row against the deployed web service.

# COMMAND ----------

import json
json_str_test_inputs = 'I crashed my car into a pole.' 
webservice.run(input_data = json_str_test_inputs)

# COMMAND ----------

# MAGIC %md # Capture the scoring URI

# COMMAND ----------

# MAGIC %md In order to call the service from a REST client, you need to acquire the scoring URI. Run the following cell to retrieve the scoring URI and take note of this value, you will need it in the last notebook.

# COMMAND ----------

webservice.scoring_uri

# COMMAND ----------

# MAGIC %md The default settings used in deploying this service result in a service that does not require authentication, so the scoring URI is the only value you need to call this service.