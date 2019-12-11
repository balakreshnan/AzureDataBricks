# Databricks notebook source
# MAGIC %md
# MAGIC # Init script to support Horovod PyTorch
# MAGIC This notebook creates an init script that installs PyTorch:
# MAGIC 
# MAGIC 1. Import this notebook to your workspace.
# MAGIC 2. Edit the `script_path` variable where you intend to save the script.
# MAGIC 3. Attach to a Python 3 cluster.
# MAGIC 4. Run this notebook (clicking **Run All** above). It will generate a script called `pytorch-init.sh` in the location you provided.
# MAGIC 
# MAGIC To create a cluster with the mount, configure a cluster with the `pytorch-init.sh` cluster-scoped init script. 

# COMMAND ----------

script_path="<script path>"
script_name = script_path + "pytorch-init.sh"

# COMMAND ----------

script='''
#!/bin/bash

set -e

/databricks/python/bin/python -V

. /databricks/conda/etc/profile.d/conda.sh

conda activate /databricks/python

conda install -y -c pytorch pytorch torchvision
'''

# COMMAND ----------

dbutils.fs.put(script_name, script, True)