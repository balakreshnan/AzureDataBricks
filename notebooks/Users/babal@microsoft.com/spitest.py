# Databricks notebook source
df = spark.read.csv('dbfs:/mnt/data/query-impala-70510.csv', header='true', inferSchema = 'true')

# COMMAND ----------

df.schema.names

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

reduceddf = df[['generated_key','ref_designator','pin_nbr','solder_paste_volume','solder_paste_height','solder_paste_area']]

# COMMAND ----------

display(reduceddf)

# COMMAND ----------

oldColumns = df.schema.names
newColumns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Range", "Volume", "Pattern", "5DayStoch", "AvgStoch", "5DaySMA", "13daySMA", "Zone", "Reccomentation"]

df = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], newColumns[idx]), xrange(len(oldColumns)), df)
df.printSchema()
df.show(5)

# COMMAND ----------

df.createOrReplaceTempView("reduceddf")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from reduceddf limit 10

# COMMAND ----------

from pyspark.sql.functions import *
split_col = split(reduceddf['generated_key'], '-')
reduceddf = reduceddf.withColumn('panel', split_col.getItem(0))
reduceddf = reduceddf.withColumn('paneldate', split_col.getItem(1))

# COMMAND ----------

display(reduceddf)

# COMMAND ----------

from ggplot import *

# COMMAND ----------

pdsDF = reduceddf.toPandas()

# COMMAND ----------

pdsDF.dtypes

# COMMAND ----------

p = ggplot(pdsDF, aes('solder_paste_area', 'solder_paste_volume')) + geom_density()

# COMMAND ----------

display(p)

# COMMAND ----------

from pyspark.sql import functions as F
display(reduceddf.groupBy("panel").agg(F.count("panel")))

# COMMAND ----------

#select only one panel 00EF4238B
eapaneldf = reduceddf.where(reduceddf.panel == '00EF4238B').collect()

# COMMAND ----------



# COMMAND ----------

display(eapaneldf)

# COMMAND ----------

display(eapaneldf[2)

# COMMAND ----------

pdseapaneldf = eapaneldf.flatMap(lambda x: x).toDF()

# COMMAND ----------

display(eapaneldf)

# COMMAND ----------

import pandas as pd
cols = reduceddf.columns
#pdseapaneldf = pd.DataFrame(np.array(eapaneldf).reshape(400,8), columns = reduceddf.columns)
pdseapaneldf = pd.DataFrame(eapaneldf, columns = reduceddf.columns)

# COMMAND ----------

reduceddf.columns

# COMMAND ----------

print(pdseapaneldf)

# COMMAND ----------

pdseapaneldf.dtypes

# COMMAND ----------

p = ggplot(pdseapaneldf, aes('pin_nbr', 'solder_paste_volume')) + geom_line()

# COMMAND ----------

display(p)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import log, isnan
import seaborn

# COMMAND ----------

colors = {'AFR': 'black', 'AMR': 'red', 'EAS': 'green', 'EUR': 'blue', 'SAS': 'cyan'}
plt.clf()
#plt.subplot(1, 1, 1)
plt.scatter(pdseapaneldf["solder_paste_volume"], pdseapaneldf["solder_paste_area"], c = pdseapaneldf["solder_paste_volume"].map(colors), alpha = .5)
plt.xlabel("PC1")
plt.ylabel("PC2")
legend_entries = [mpatches.Patch(color= c, label=pheno) for pheno, c in colors.items()]
plt.legend(handles=legend_entries)
plt.show()
display()

# COMMAND ----------

plt.clf()
plt.plot(pdseapaneldf["solder_paste_height"],pdseapaneldf["solder_paste_volume"])
plt.show()
display()

# COMMAND ----------

plt.clf()
plt.figure(1)
plt.subplot(211)
plt.plot(pdseapaneldf["solder_paste_volume"])
plt.subplot(212)
plt.plot(pdseapaneldf["solder_paste_area"])
plt.show()
display()

# COMMAND ----------

pdseapaneldf.index.values

# COMMAND ----------

plt.clf()
#xticks = pdseapaneldf['pin_bnr']

plt.bar(pdseapaneldf.index.values,pdseapaneldf["solder_paste_volume"],1.0)
plt.show()
display()

# COMMAND ----------

fig, ax = pdseapaneldf.plot(kind='bar',x='pin_nbr',y='solder_paste_volume',colormap='winter_r')
#ax.plot(x, y, 'k--')

# COMMAND ----------

display(fig)

# COMMAND ----------

from pandas.tools.plotting import scatter_matrix
stuff = scatter_matrix(pdsDF, alpha=0.7, figsize=(6, 6), diagonal='kde')

# COMMAND ----------

display(stuff)

# COMMAND ----------

print(pdsDF)

# COMMAND ----------

plt.clf()
plt.plot(pdsDF["solder_paste_volume"])
plt.show()
display()

# COMMAND ----------

import spark.implicits._
val collection = aggdf1.collect().slice(2000,3000)
val distDataRDD = sc.parallelize(collection)

val schema = new StructType()
  .add(StructField("generated_key", StringType, true))
  .add(StructField("pin_nbr", StringType, true))
  .add(StructField("paste_volume_average", DoubleType, true))

val df = spark.createDataFrame(distDataRDD, schema)
display(df) 