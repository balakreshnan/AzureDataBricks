# Databricks notebook source
# MAGIC %md 
# MAGIC # Distributed model inference using PyTorch
# MAGIC This notebook demonstrates how to do distributed model inference using PyTorch with ResNet-50 model and image files as input data.
# MAGIC 
# MAGIC This guide consists of the following sections:
# MAGIC 
# MAGIC * **Prepare trained model and data for inference.**
# MAGIC   * Load pre-trained ResNet-50 model from [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50).
# MAGIC   * Downloads the Flowers data to the shared space.
# MAGIC * **Load the data into Spark DataFrames.** 
# MAGIC * **Run model inference via Pandas UDF.** 
# MAGIC 
# MAGIC **Note:**
# MAGIC * To run the notebook on CPU-enabled Apache Spark clusters, change the variable `cuda = False`.
# MAGIC * To run the notebook on GPU-enabled Apache Spark clusters, change the variable `cuda = True`.
# MAGIC * To run the notebook, create a cluster with Databricks Runtime 5.0 ML or above.

# COMMAND ----------

cuda = False

# COMMAND ----------

# Enable Arrow support.
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "2048")

# COMMAND ----------

import os
import shutil
import tarfile
import time
import zipfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader  # private API

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType

# COMMAND ----------

use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Prepare trained model and data for inference

# COMMAND ----------

# MAGIC %md
# MAGIC Define the input and output directory.

# COMMAND ----------

fuse_location = "<FUSE_MOUNT_LOCATION>" 
URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
input_local_dir = "/mnt/{}/tmp/".format(fuse_location)
output_file_path = "/tmp/predictions"

# COMMAND ----------

# MAGIC %md
# MAGIC Load ResNet50 on driver node and broadcast its state.

# COMMAND ----------

bc_model_state = sc.broadcast(models.resnet50(pretrained=True).state_dict())

# COMMAND ----------

def get_model_for_eval():
  """Gets the broadcasted model."""
  model = models.resnet50(pretrained=True)
  model.load_state_dict(bc_model_state.value)
  model.eval()
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC Download the Google Flowers dataset. It might take a while.

# COMMAND ----------

def maybe_download_and_extract(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            
        file_path, _ = urlretrieve(url=url, filename=file_path)
        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")

# COMMAND ----------

maybe_download_and_extract(url=URL, download_dir=input_local_dir)

# COMMAND ----------

local_dir = input_local_dir + 'flower_photos/'
files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(local_dir) for f in filenames if os.path.splitext(f)[1] == '.jpg']
len(files)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the data into Spark DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC Create a DataFrame of image paths.

# COMMAND ----------

files_df = spark.createDataFrame(
  map(lambda path: (path,), files), ["path"]
).repartition(10)  # number of partitions should be a small multiple of total number of nodes
display(files_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run model inference via Pandas UDF

# COMMAND ----------

# MAGIC %md
# MAGIC Create a custom PyTorch dataset class.

# COMMAND ----------

class ImageDataset(Dataset):
  def __init__(self, paths, transform=None):
    self.paths = paths
    self.transform = transform
  def __len__(self):
    return len(self.paths)
  def __getitem__(self, index):
    image = default_loader(self.paths[index])
    if self.transform is not None:
      image = self.transform(image)
    return image

# COMMAND ----------

# MAGIC %md
# MAGIC Define the function for model inference.

# COMMAND ----------

def predict_batch(paths):
  transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
  ])
  images = ImageDataset(paths, transform=transform)
  loader = torch.utils.data.DataLoader(images, batch_size=500, num_workers=8)
  model = get_model_for_eval()
  model.to(device)
  all_predictions = []
  with torch.no_grad():
    for batch in loader:
      predictions = list(model(batch.to(device)).cpu().numpy())
      for prediction in predictions:
        all_predictions.append(prediction)
  return pd.Series(all_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC Test the function locally.

# COMMAND ----------

predictions = predict_batch(pd.Series(files[:200]))

# COMMAND ----------

# MAGIC %md 
# MAGIC Wrap the function as a Pandas UDF.

# COMMAND ----------

predict_udf = pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)(predict_batch)

# COMMAND ----------

# MAGIC %md
# MAGIC Run the model inference and save the result to a Parquet file.

# COMMAND ----------

# Make predictions.
predictions_df = files_df.select(col('path'), predict_udf(col('path')).alias("prediction"))
predictions_df.write.mode("overwrite").parquet(output_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Load and check the prediction results.

# COMMAND ----------

result_df = spark.read.load(output_file_path)
display(result_df)

# COMMAND ----------

dbutils.fs.ls(input_local_dir)