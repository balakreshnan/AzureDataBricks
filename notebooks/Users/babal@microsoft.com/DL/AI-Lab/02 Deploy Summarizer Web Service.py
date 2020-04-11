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
# MAGIC * Library Source: PyPi
# MAGIC * Package: `azureml-sdk[databricks]`
# MAGIC * Select Create
# MAGIC 
# MAGIC **NOTE: It takes a few minutes to install the Azure ML SDK, so be sure the library has completed installation before proceeding**

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

# MAGIC %md #Initialize Azure ML Workspace
# MAGIC 
# MAGIC In this notebook, you will use the Azure Machine Learning SDK to create a new Azure Machine Learning Workspace in your Azure Subscription.
# MAGIC 
# MAGIC Please specify the Azure subscription Id, resource group name, workspace name, and the region in which you want to create the Azure Machine Learning Workspace. 
# MAGIC 
# MAGIC You can get the value of your Azure subscription ID from the Azure Portal, and then selecting Subscriptions from the menu on the left.
# MAGIC 
# MAGIC For the `resource_group`, use the name of the resource group that contains your Azure Databricks Workspace. 
# MAGIC 
# MAGIC NOTE: If you provide a resource group name that does not exist, the resource group will be automatically created. This may or may not succeed in your environment, depending on the permissions you have on your Azure Subscription.

# COMMAND ----------

#Provide the Subscription ID of your existing Azure subscription
subscription_id = "e223f1b3-d19b-4cfa-98e9-bc9be62717bc"#"<you-azure-subscription-id>"

#Provide a name for the Resource Group that will contain Azure ML related services 
resource_group = "mcwailab"

# Provide the name and region for the Azure Machine Learning Workspace that will be created
workspace_name = "azureml" 
workspace_region = "westcentralus"#'eastus2' # eastus, westcentralus, southeastasia, australiaeast, westeurope

# COMMAND ----------

# MAGIC %md #Create an Azure ML Workspace

# COMMAND ----------

# MAGIC %md Run the following cell and follow the instructions printed in the output. 
# MAGIC 
# MAGIC You will see instructions that read:
# MAGIC 
# MAGIC `Performing interactive authentication. Please follow the instructions on the terminal.`
# MAGIC 
# MAGIC `To sign in, use a web browser to open the page https://microsoft.com/devicelogin and enter the code SOMECODE to authenticate.`
# MAGIC 
# MAGIC When you see this, open a new browser window, navigate to the provided URL. At the code prompt, enter the code provided (be sure to delete any trailing spaces).
# MAGIC 
# MAGIC Login with the same credentials you use to access your Azure subscription.
# MAGIC 
# MAGIC Once you have authenticated, the output will continue.
# MAGIC 
# MAGIC When you see `Provisioning complete.` your Workspace has been created and you can move on to the next cell. 

# COMMAND ----------

import azureml.core

# import the Workspace class and check the azureml SDK version
from azureml.core import Workspace

ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region)

print("Provisioning complete.")

# COMMAND ----------

# MAGIC %md #Persist the Workspace configuration

# COMMAND ----------

# MAGIC %md Run the following cells to retrieve the configuration of the deployed Workspace and persist it to local disk and then to the Databricks Filesystem.

# COMMAND ----------

import os
import shutil
import azureml.core
# import the Workspace class and check the azureml SDK version
from azureml.core import Workspace

ws = Workspace(
    workspace_name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group)

# persist the subscription id, resource group name, and workspace name in aml_config/config.json.
aml_config = 'aml_config'
if os.path.isfile(aml_config) or os.path.isdir(aml_config):
    shutil.rmtree(aml_config)
ws.write_config()

# COMMAND ----------

# MAGIC %md 
# MAGIC Take a look at the contents of the generated configuration file by running the following cell:

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /databricks/driver/aml_config/config.json

# COMMAND ----------

# MAGIC %md
# MAGIC Copy the config file to DBFS

# COMMAND ----------

#persist the config file to dbfs so that it can be used for the other notebooks.
aml_config_local = 'file:' + os.getcwd() + '/' + aml_config
aml_config_dbfs = '/dbfs/' + 'aml_config'

if os.path.isfile(aml_config_dbfs) or os.path.isdir(aml_config_dbfs):
    shutil.rmtree(aml_config_dbfs)
    #dbutils.fs.rm(aml_config, recurse=True)

dbutils.fs.cp(aml_config_local, aml_config, recurse=True)

# COMMAND ----------

# MAGIC %md #Deploy model to Azure Container Instance (ACI)

# COMMAND ----------

# MAGIC %md In this section, you will deploy a web service that uses Gensim as shown above to summarize text. The web service will be hosted in Azure Container Service.

# COMMAND ----------

import azureml.core
from azureml.core.workspace import Workspace

#get the config file from dbfs
aml_config = '/aml_config'
dbutils.fs.cp(aml_config, 'file:'+os.getcwd()+aml_config, recurse=True)

ws = Workspace.from_config()

# COMMAND ----------

# MAGIC %md ##Create the scoring web service

# COMMAND ----------

# MAGIC %md When deploying models for scoring with Azure Machine Learning services, you need to define the code for a simple web service that will load your model and use it for scoring. By convention this service has two methods `init` which loads the model and `run` which scores data using the loaded model. 
# MAGIC 
# MAGIC This scoring service code will later be deployed inside of a specially prepared Docker container.

# COMMAND ----------

#%%writefile summarizer_service.py
summarizer_service = """

import re
import nltk
import unicodedata
from gensim.summarization import summarize, keywords

def clean_and_parse_document(document):
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')
    else:
        raise ValueError("Document is not string or unicode.")
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences

def summarize_text(text, summary_ratio=None, word_count=30):
    sentences = clean_and_parse_document(text)
    cleaned_text = ' '.join(sentences)
    summary = summarize(cleaned_text, split=True, ratio=summary_ratio, word_count=word_count)
    return summary 

def init():  
    nltk.download('all')
    return

def run(input_str):
    try:
        return summarize_text(input_str)
    except Exception as e:
        return (str(e))
"""

exec(summarizer_service)

with open("summarizer_service.py", "w") as file:
    file.write(summarizer_service)
    print("Summarizer service saved as summarizer_service.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Conda dependencies environment file

# COMMAND ----------

# MAGIC %md Your web service can have dependencies installed by using a Conda environment file. Items listed in this file will be conda or pip installed within the Docker container that is created and thus be available to your scoring web service logic.

# COMMAND ----------

from azureml.core.conda_dependencies import CondaDependencies 

myacienv = CondaDependencies.create(pip_packages=['gensim','nltk'])

with open("mydeployenv.yml","w") as f:
    f.write(myacienv.serialize_to_string())

# COMMAND ----------

# MAGIC %md ##Deployment

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
    tags = {'name':'Summarization'}, 
    description = 'Summarizes text.')

# COMMAND ----------

# MAGIC %md Next, build up a container image configuration that names the scoring service script, the runtime (python or Spark), and provides the conda file.

# COMMAND ----------

service_name = "summarizer"
runtime = "python" #"python" #
driver_file = "summarizer_service.py"
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

webservice = Webservice.deploy(
  workspace=ws, 
  name=service_name, 
  model_paths=[],
  deployment_config=aci_config,
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

example_document = """
I was driving down El Camino and stopped at a red light.
It was about 3pm in the afternoon.  
The sun was bright and shining just behind the stoplight.
This made it hard to see the lights.
There was a car on my left in the left turn lane.
A few moments later another car, a black sedan pulled up behind me. 
When the left turn light changed green, the black sedan hit me thinking 
that the light had changed for us, but I had not moved because the light 
was still red.
After hitting my car, the black sedan backed up and then sped past me.
I did manage to catch its license plate. 
The license plate of the black sedan was ABC123. 
"""

# COMMAND ----------

import json
json_str_test_inputs = json.dumps(example_document)
result = webservice.run(input_data = json_str_test_inputs)
print(result)

# COMMAND ----------

# MAGIC %md # Capture the scoring URI

# COMMAND ----------

# MAGIC %md In order to call the service from a REST client, you need to acquire the scoring URI. Run the following cell to retrieve the scoring URI and take note of this value, you will need it in the last notebook.

# COMMAND ----------

webservice.scoring_uri

# COMMAND ----------

# MAGIC %md The default settings used in deploying this service result in a service that does not require authentication, so the scoring URI is the only value you need to call this service.