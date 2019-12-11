// Databricks notebook source
// MAGIC %md #### Create an [Init Script](https://docs.databricks.com/user-guide/clusters/init-scripts.html) for the cluster you want to install XGBoost on
// MAGIC In this example, we create an init script for the cluster named 'xgboost'. The cluster name is important here. 

// COMMAND ----------

dbutils.fs.put("/databricks/init/xgboost/install-xgboost.sh", """
#!/bin/bash 
sudo apt-get -y install git
sudo apt-get update
sudo apt-get install -y maven
sudo apt-get install -y cmake

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
git clone --recursive https://github.com/dmlc/xgboost

cd xgboost/jvm-packages
mvn -DskipTests package 

cp xgboost4j-spark/target/xgboost4j-spark-*-jar-with-dependencies.jar /databricks/jars/
""", true)

// COMMAND ----------

// MAGIC %md ####Restart/create the cluster named 'xgboost'
// MAGIC This executes the script we just created, which will install XGBoost on each cluster node and move the .jar file into the correct location