# Databricks notebook source
# MAGIC %md <h1>Creating a searchable Art Database with The MET's open-access collection</h1>

# COMMAND ----------

# MAGIC %md In this example, we show how you can enrich data using Cognitive Skills and write to an Azure Search Index using MMLSpark. We use a subset of The MET's open-access collection and enrich it by passing it through 'Describe Image' and a custom 'Image Similarity' skill. The results are then written to a searchable index.

# COMMAND ----------

import numpy as np, pandas as pd, os, sys, time, json, requests

from mmlspark import *
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType, StringType, DoubleType, StructType, StructField, ArrayType
from pyspark.ml import Transformer, Estimator, Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.sql.functions import lit, udf, col, split

# COMMAND ----------

VISION_API_KEY = os.environ['VISION_API_KEY']
AZURE_SEARCH_KEY = os.environ['AZURE_SEARCH_KEY']
search_service = "mmlspark-azure-search"
search_index = "test"

# COMMAND ----------

data = spark.read\
  .format("csv")\
  .option("header", True)\
  .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/metartworks_sample.csv")\
  .withColumn("searchAction", lit("upload"))\
  .withColumn("Neighbors", split(col("Neighbors"), ",").cast("array<string>"))\
  .withColumn("Tags", split(col("Tags"), ",").cast("array<string>"))\
  .limit(25)

# COMMAND ----------

# MAGIC %md <img src="https://mmlspark.blob.core.windows.net/graphics/CognitiveSearchHyperscale/MetArtworkSamples.png" width="800" style="float: center;"/>

# COMMAND ----------

#define pipeline
describeImage = DescribeImage()\
                .setSubscriptionKey(VISION_API_KEY)\
                .setUrl("https://eastus.api.cognitive.microsoft.com/vision/v2.0/describe")\
                .setImageUrlCol("PrimaryImageUrl")\
                .setOutputCol("RawImageDescription")\
                .setConcurrency(5)\
                .setMaxCandidates(5)

getDescription = SQLTransformer(statement="SELECT *, RawImageDescription.description.captions.text as ImageDescriptions \
                                FROM __THIS__")

getTags = SQLTransformer(statement="SELECT *, RawImageDescription.description.tags as ImageTags FROM __THIS__")
                
finalcols = SelectColumns().setCols(['ObjectID', 'Department', 'Culture', 'Medium', 'Classification', 'PrimaryImageUrl',\
                                     'Tags', 'Neighbors', 'ImageDescriptions', 'searchAction'])

data_processed = Pipeline(stages = [describeImage, getDescription, getTags, finalcols])\
                    .fit(data).transform(data)

# COMMAND ----------

# MAGIC %md <img src="https://mmlspark.blob.core.windows.net/graphics/CognitiveSearchHyperscale/MetArtworksProcessed.png" width="800" style="float: center;"/>

# COMMAND ----------

# MAGIC %md Before writing the results to a Search Index, you must define a schema which must specify the name, type, and attributes of each field in your index. Refer [Create a basic index in Azure Search](https://docs.microsoft.com/en-us/azure/search/search-what-is-an-index) for more information.

# COMMAND ----------

index_dict = { "name" : search_index, 
               "fields" : [
                  {
                       "name": "ObjectID", 
                       "type": "Edm.String", 
                       "key": True, 
                       "facetable": False
                  },
                  {
                       "name": "Department", 
                       "type": "Edm.String", 
                       "facetable": False
                  }, 
                  {
                       "name": "Culture", 
                       "type": "Edm.String", 
                       "facetable": False
                  },
                  {
                       "name": "Medium", 
                       "type": "Edm.String", 
                       "facetable": False
                  },
                  {
                       "name": "Classification", 
                       "type": "Edm.String", 
                       "facetable": False
                  },
                  {
                       "name": "PrimaryImageUrl", 
                       "type": "Edm.String", 
                       "facetable": False
                  },                   
                  {
                       "name": "Tags", 
                       "type": "Collection(Edm.String)", 
                       "facetable": False
                  },                  
                  {
                       "name": "Neighbors", 
                       "type": "Collection(Edm.String)", 
                       "facetable": False
                  },
                  {
                       "name": "ImageDescriptions", 
                       "type": "Collection(Edm.String)", 
                       "facetable": False
                  }
              ]}

index_str = json.dumps(index_dict)

options = {
            "subscriptionKey" : AZURE_SEARCH_KEY,
            "actionCol" : "searchAction",
            "serviceName" : search_service,
            "indexJson" : index_str
          }

# COMMAND ----------

data_processed.writeToAzureSearch(options)

# COMMAND ----------

# MAGIC %md The Search Index can be queried using the [Azure Search REST API](https://docs.microsoft.com/rest/api/searchservice/) by sending GET or POST requests and specifying query parameters that give the criteria for selecting matching documents. For more information on querying refer [Query your Azure Search index using the REST API](https://docs.microsoft.com/en-us/rest/api/searchservice/Search-Documents)

# COMMAND ----------

post_url = 'https://%s.search.windows.net/indexes/%s/docs/search?api-version=2017-11-11' % (search_service, search_index)

headers = {
    "Content-Type":"application/json",
    "api-key": os.environ['AZURE_SEARCH_KEY']
}

# COMMAND ----------

body = {
    "search": "Glass",
    "searchFields": "Classification",
    "select": "ObjectID, Department, Culture, Medium, Classification, PrimaryImageUrl, Tags, ImageDescriptions",
    "top":"3"
}

# COMMAND ----------

response = requests.post(post_url, json.dumps(body), headers = headers)

# COMMAND ----------

response.json()