# Databricks notebook source
# Import libraries and display version
import statsmodels
print("Using statsmodels version: ", statsmodels.__version__)
import pandas
print("Using pandas version: ", pandas.__version__)
import numpy
print("Using numpy version: ", numpy.__version__)
import tsfresh
print("Using tsfresh version: ", tsfresh.__version__)
import sklearn
print("Using sklearn version: ", sklearn.__version__)
import matplotlib
print("Using matplotlib version: ", matplotlib.__version__)
import skfuzzy
print("Using skfuzzy version: ", skfuzzy.__version__)
import pyspark
print("Using pyspark version: ", pyspark.__version__)

# Import libraries that do not have a __version__ attribute
import time
import random
import itertools
import spark_sklearn
import category_encoders
import pickle

# Import specialized packages within libraries
from sklearn import ensemble, metrics, model_selection, neighbors, preprocessing, svm, tree, naive_bayes
from sklearn.metrics import classification_report, confusion_matrix
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
from random import randint
import matplotlib.pylab as plt

# Set random seed
seed = 7
numpy.random.seed(seed)

# COMMAND ----------

# Load consolidated DF from .parquet file
# importedSparkDF = spark.read.parquet("/tmp/BGA_InputData_Updated.parquet")  # This adds columns to help output / interpretation

# Load data from individual .csv files in FileStore as Spark DataFrames

# inputSparkDF_114336_2T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_114336_2T-a8d65.csv')
# inputSparkDF_114336_2T_MAY = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_114336_2T_May-8c6df.csv')
# inputSparkDF_114336_2T_JUN = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_114336_2T_Jun-27d87.csv')
# inputSparkDF_114336_2T_OCT = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/114336_2T_TestOctober2018-12c60.csv')
# inputSparkDF_200188_4T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_200188_4T-3bbd1.csv')
# inputSparkDF_200189_4T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_200189_4T_Part1-1743d.csv') 
# inputSparkDF_200189_4T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_200189_4T_Part2-d931b.csv')
# inputSparkDF_200189_4T_Part3 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_200189_4T-a61a0.csv')
# inputSparkDF_221364_1T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_221364_1T_Part1-4337f.csv')
# inputSparkDF_221364_1T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_221364_1T_Part2-34c2e.csv')
# inputSparkDF_264469_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_264469_1T-0f7f5.csv')
# inputSparkDF_302139_2T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_302139_2T-f7dc2.csv')
# inputSparkDF_337123_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_337123_1T-c1789.csv')
# inputSparkDF_337123_1T_MAR = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_337123_1T_Mar-9998d.csv')
inputSparkDF_337123_1T_APR = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_337123_1T_Apr-7b0c0.csv')
# inputSparkDF_337123_1T_MAY = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_337123_1T_May-10ab1.csv')
# inputSparkDF_337123_1T_JUN = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_337123_1T_Jun-faa7e.csv')
# inputSparkDF_337123_1T_JUL = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_337123_1T_Jul-17217.csv')
# inputSparkDF_337123_1T_AUG = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_337123_1T_Aug-cb233.csv')
# inputSparkDF_337123_1T_SEP = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_337123_1T_Sep-2fde2.csv')
# inputSparkDF_337123_1T_OCT = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_337123_1T_Oct-ed999.csv')
# inputSparkDF_339721_1T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_339721_1T_Part1-5d7d5.csv')
# inputSparkDF_339721_1T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_339721_1T_Part2-da85a.csv')
# inputSparkDF_339723_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_339723_1T-46362.csv')
# inputSparkDF_351299_1T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_351299_1T_Part1-fd4eb.csv')
# inputSparkDF_351299_1T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_351299_1T_Part2-e854a.csv')
# inputSparkDF_358387_2T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_358387_2T-2390b.csv')
# inputSparkDF_375220_2T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_375220_2T-19efe.csv')
# inputSparkDF_375459_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_375459_1T-e2ab9.csv')
# inputSparkDF_386238_1T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_386238_1T_Part1-5bb86.csv')
# inputSparkDF_386238_1T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_386238_1T_Part2-5abff.csv')
# inputSparkDF_420084_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_420084_1T-8432d.csv')
# inputSparkDF_454408_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_454408_1T-c5fe8.csv')
# inputSparkDF_474138_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_474138_1T-6a500.csv')
# inputSparkDF_478000_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_478000_1T-d7f61.csv')

# COMMAND ----------

# print((importedSparkDF.count(), len(importedSparkDF.columns)))
# print((inputSparkDF_454408_1T.count(), len(inputSparkDF_454408_1T.columns)))
# print((inputSparkDF_474138_1T.count(), len(inputSparkDF_474138_1T.columns)))
# print((inputSparkDF_478000_1T.count(), len(inputSparkDF_478000_1T.columns)))

print((inputSparkDF_337123_1T_APR.count(), len(inputSparkDF_337123_1T_APR.columns)))
# print((inputSparkDF_221364_1T_Part2.count(), len(inputSparkDF_221364_1T_Part2.columns)))
# print((inputSparkDF_200189_4T_Part3.count(), len(inputSparkDF_200189_4T_Part3.columns)))

# COMMAND ----------

# inputData = spark_sklearn.Converter(importedSparkDF.toPandas())
# inputData = importedSparkDF.toPandas()
# inputData = inputData.drop_duplicates()

# merged = inputSparkDF_221364_1T_Part1.unionAll(inputSparkDF_221364_1T_Part2)
# merged2 = merged.unionAll(inputSparkDF_200189_4T_Part3)
# inputData = merged2.toPandas()
inputData = inputSparkDF_337123_1T_APR.toPandas()


# COMMAND ----------

print(inputData['fndng_area_cd'].unique())

# COMMAND ----------

# Create columns for binary response variable indicating an AOI defect
conditions_aoi = [inputData['fndng_area_cd'].str.contains("_SM", na=False)]
conditions_ict = [(inputData['fndng_area_cd'].str.contains("_ICT", na=False)), (inputData['fndng_area_cd'].str.contains("_MAN", na=False))]
conditions_fa = [(inputData['fndng_area_cd'].str.contains("_FINAL", na=False)), (inputData['fndng_area_cd'].str.contains("_RAP", na=False))]
choices_aoi = [1]
choices_ict = [1, 1]
choices_fa = [1, 1]
inputData['aoi_defect'] = numpy.select(conditions_aoi, choices_aoi, default=0)
inputData['ict_defect'] = numpy.select(conditions_ict, choices_ict, default=0)
inputData['fa_defect'] = numpy.select(conditions_fa, choices_fa, default=0)
inputData['any_defect'] = inputData['aoi_defect'] + inputData['ict_defect'] + inputData['fa_defect']
print(inputData['aoi_defect'].sum())
print(inputData['ict_defect'].sum())
print(inputData['fa_defect'].sum())
print(inputData['any_defect'].sum())

"""
inputData['AOI'] = numpy.select(conditions_aoi, choices_aoi, default=0)
inputData['ICT'] = numpy.select(conditions_ict, choices_ict, default=0)
inputData['FA'] = numpy.select(conditions_fa, choices_fa, default=0)
inputData['ANY'] = inputData['AOI'] + inputData['ICT'] + inputData['FA']
print(inputData['AOI'].sum(), inputData['aoi_defect'].sum())
print(inputData['ICT'].sum(), inputData['ict_defect'].sum())
print(inputData['FA'].sum(), inputData['fa_defect'].sum())
print(inputData['ANY'].sum(), inputData['any_defect'].sum())
"""

# COMMAND ----------

"""
inputData["AOI_Diff"] = inputData['AOI'] - inputData['aoi_defect']
inputData["ICT_Diff"] = inputData['ICT'] - inputData['ict_defect']
inputData["FA_Diff"] = inputData['FA'] - inputData['fa_defect']
inputData["ANY_Diff"] = inputData['ANY'] - inputData['any_defect']
"""
"""
print((inputData['AOI'] - inputData['aoi_defect']).sum())
print((inputData['ICT'] - inputData['ict_defect']).sum())
print((inputData['FA'] - inputData['fa_defect']).sum())
print((inputData['ANY'] - inputData['any_defect']).sum())
"""

# COMMAND ----------

inputData["BarcodeSilkscreenRefID"] = inputData['barcd'].map(str) + "_" + inputData['silkscrn_nbr'].map(str) + "_" + inputData['ref_id']

# COMMAND ----------

print(inputData.shape)

# COMMAND ----------

# Sort by Barcode_Silkscreen_Reference Designator, then Component Pin Number, then presence of any defect
inputData = inputData.sort_values(['BarcodeSilkscreenRefID', 'cmpntpin_nbr', 'ict_defect'])

# COMMAND ----------

# Create a DataFrame for numeric input data to model as time-series-like within the barcode-silkscreen-reference_id index
numericInputData = pandas.DataFrame()
numericInputData['id'] = inputData['BarcodeSilkscreenRefID']
numericInputData['pin'] = inputData['cmpntpin_nbr']
numericInputData['sp_o_x'] = inputData['solder_paste_offst_x_axis']
numericInputData['sp_o_y'] = inputData['solder_paste_offst_y_axis']
numericInputData['sp_v_p'] = inputData['solder_paste_vol_pct']
numericInputData['sp_h'] = inputData['solder_paste_hgt']
numericInputData['sp_a_p'] = inputData['solder_paste_area_pct']
numericInputData['aoi_defect'] = inputData['aoi_defect']
numericInputData['ict_defect'] = inputData['ict_defect']
numericInputData['fa_defect'] = inputData['fa_defect']
numericInputData['any_defect'] = inputData['any_defect']



# COMMAND ----------

numericInputData_red1 = numericInputData.drop_duplicates(keep='last')
numericInputData_red1b = numericInputData.drop_duplicates(subset=['id', 'pin'], keep='last')

# COMMAND ----------

print(numericInputData.duplicated(keep='last').sum())
print(numericInputData_red1.duplicated(keep='last').sum())

# COMMAND ----------

print(numericInputData_red1.shape)
print(numericInputData_red1b.shape)

# COMMAND ----------

print(numericInputData_red1['id'].nunique())
print(numericInputData_red1b['id'].nunique())

# COMMAND ----------

# Create a DataFrame for each response variable
responseAOI = pandas.DataFrame()
responseAOI['id'] = numericInputData_red1b['id']
responseAOI['aoi_defect'] = numericInputData_red1b['aoi_defect']

responseICT = pandas.DataFrame()
responseICT['id'] = numericInputData_red1b['id']
responseICT['ict_defect'] = numericInputData_red1b['ict_defect']

responseFA = pandas.DataFrame()
responseFA['id'] = numericInputData_red1b['id']
responseFA['fa_defect'] = numericInputData_red1b['fa_defect']

# COMMAND ----------

print(numericInputData_red1b.shape)
print(responseAOI.shape)
print(responseICT.shape)
print(responseFA.shape)

# COMMAND ----------

responseAOI = responseAOI.sort_values(['id', 'aoi_defect'])
responseICT = responseICT.sort_values(['id', 'ict_defect'])
responseFA = responseFA.sort_values(['id', 'fa_defect'])

# COMMAND ----------

print(responseAOI.duplicated(keep=False).sum())
print(responseICT.duplicated(keep=False).sum())
print(responseFA.duplicated(keep=False).sum())
print(responseAOI.duplicated(keep='last').sum())
print(responseICT.duplicated(keep='last').sum())
print(responseFA.duplicated(keep='last').sum())

# COMMAND ----------

"""
Remove duplicate records - first step:  Keep all values EXCEPT ("~") the last duplicate. 
"""

# responseAOI_red1 = responseAOI[~(responseAOI.duplicated(keep='last'))]
# responseICT_red1 = responseICT[~(responseICT.duplicated(keep='last'))]
# responseFA_red1 = responseFA[~(responseFA.duplicated(keep='last'))]

responseAOI_red1 = responseAOI.drop_duplicates(keep='last')
responseICT_red1 = responseICT.drop_duplicates(keep='last')
responseFA_red1 = responseFA.drop_duplicates(keep='last')

# COMMAND ----------

print(responseAOI_red1.shape)
print(responseICT_red1.shape)
print(responseFA_red1.shape)

# COMMAND ----------


responseAOI_red2 = responseAOI_red1[~((responseAOI_red1.duplicated(subset=['id'], keep=False)) & (responseAOI_red1['aoi_defect'] == 0))]
responseICT_red2 = responseICT_red1[~((responseICT_red1.duplicated(subset=['id'], keep=False)) & (responseICT_red1['ict_defect'] == 0))]
responseFA_red2 = responseFA_red1[~((responseFA_red1.duplicated(subset=['id'], keep=False)) & (responseFA_red1['fa_defect'] == 0))]

# COMMAND ----------

# Print shape of each response variable DataFrame - confirm the same number of records as unique indices in numericInputData

print(responseAOI_red2.shape)
print(responseICT_red2.shape)
print(responseFA_red2.shape)

# COMMAND ----------

# Display number of records (Barcode_Silkscreen_RefID) with an identified defect
print(responseAOI['aoi_defect'].sum())
print(responseAOI_red1['aoi_defect'].sum())
print(responseAOI_red2['aoi_defect'].sum())

print(responseICT['ict_defect'].sum())
print(responseICT_red1['ict_defect'].sum())
print(responseICT_red2['ict_defect'].sum())

print(responseFA['fa_defect'].sum())
print(responseFA_red1['fa_defect'].sum())
print(responseFA_red2['fa_defect'].sum())

# COMMAND ----------

# Convert response variable to Spark DataFrame, if so desired, for purposes of saving to .parquet file
responseAOI_spark = spark.createDataFrame(responseAOI_red2)
responseICT_spark = spark.createDataFrame(responseICT_red2)
responseFA_spark = spark.createDataFrame(responseFA_red2)

# COMMAND ----------

# Response variable must be in the form of a Pandas.Series in order for TSFRESH to work
responseAOI_tsfresh = pandas.Series(responseAOI_red2.values[:, 1], index=responseAOI_red2.id)
responseICT_tsfresh = pandas.Series(responseICT_red2.values[:, 1], index=responseICT_red2.id)
responseFA_tsfresh = pandas.Series(responseFA_red2.values[:, 1], index=responseFA_red2.id)

# COMMAND ----------

# Generate features from "TS" data using TSFRESH package

extraction_settings=ComprehensiveFCParameters()  # Generate all possible features
# extraction_settings=EfficientFCParameters()  # Generate only features with "efficient" processing time
# extraction_settings=MinimalFCParameters()  # Generate minimal, quickly-generated features

X = extract_features(numericInputData_red1b.iloc[:, 0:7], column_id='id', column_sort='pin', default_fc_parameters=extraction_settings, impute_function=impute)
# X = extract_features(numericInputData_red1b.iloc[:, [0, 1, 4, 5, 6]], column_id='id', column_sort='pin', default_fc_parameters=extraction_settings, impute_function=impute)

# COMMAND ----------

# Create Spark dataframe of matrix of extracted features
# X_spark = spark.createDataFrame(X)


# COMMAND ----------

# Display Spark dataframe of matrix of extracted features - download to .csv if so desired
# display(X_spark)

# COMMAND ----------

# Display Spark DataFrame for response variable - download to .csv if so desired
# display(responseICT_spark)

# COMMAND ----------

# Split data into training set and test set
X_placeholder, X_test, y_placeholder, y_test = model_selection.train_test_split(X, responseICT_tsfresh, test_size = 0.2, random_state=2, shuffle=True)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_placeholder, y_placeholder, test_size = 0.25, random_state=2, shuffle=True)

# COMMAND ----------

print(y_placeholder.sum())
print(y_train.sum())
print(y_val.sum())
print(y_test.sum())
print(X_train.shape[1])
print(X_placeholder.shape[1])

# COMMAND ----------

    # Encode categorical variables - replace category levels with numeric values (Should not be necessary if data is exclusively TSFRESH extracted features)
    encoder = category_encoders.OrdinalEncoder()  # Not *best* but pretty close and obtained quickly
    # encoder = category_encoders.LeaveOneOutEncoder()  # Similar to target encoding; best overall result
    X_placeholder_E = encoder.fit_transform(X_placeholder, y_placeholder)
    X_train_E = encoder.fit_transform(X_train, y_train)  # Fit the training set with the encodings
    X_val_E = encoder.transform(X_val)  # Convert categories in validation set using the encoding previously fitted
    X_test_E = encoder.transform(X_test)  # Convert categories in test set using the encoding previously fitted

# COMMAND ----------

# Filter #1 - Reduce Features Using TSFRESH
# reduced_feature_set = select_features(X_train_E, y_train)
reduced_feature_set = select_features(X_placeholder_E, y_placeholder)
remaining_column_names = reduced_feature_set.columns.values  # Names of remaining columns

# Relabel X_train for use below
# X_train_Filter1 = reduced_feature_set
X_placeholder_Filter1 = reduced_feature_set

# Ensure that validation set columns correspond to training set columns
X_val_Filter1 = X_val_E[remaining_column_names]

# Ensure that test set columns correspond to training set columns
X_test_Filter1 = X_test_E[remaining_column_names]

# print("Number of features remaining in model: ", '\t', X_train.shape[1])
# print("Column names of features remaining in model: ", '\n', remaining_columns, '\n')

# COMMAND ----------

# print("Number of features remaining in model: ", '\t', X_train_Filter1.shape[1])
print("Number of features remaining in model: ", '\t', X_placeholder_Filter1.shape[1])
print("Column names of features remaining in model: ", '\n', remaining_column_names, '\n')

# COMMAND ----------

# Create Spark dataframe of matrix of extracted / filtered features
#sparkDF = spark.createDataFrame(X_placeholder)

# COMMAND ----------

# Display Spark dataframe of matrix of extracted features - download to .csv if so desired
#display(sparkDF)

# COMMAND ----------

# Train model - Select only one of the options below
model = tree.DecisionTreeClassifier(random_state=seed) # Decision Tree
# model = tree.DecisionTreeClassifier(class_weight={0:1, 1:10}, random_state=seed) # Decision Tree
# model = ensemble.RandomForestClassifier(random_state=seed, n_estimators=100)  # Random Forest
# model = neighbors.KNeighborsClassifier()  # K-Nearest Neighbor
# model = svm.SVC()  # Support Vector Machines
# model = naive_bayes.GaussianNB()

# Fit model 
model.fit(X_placeholder_Filter1, y_placeholder.astype(int))  # Use this to train model on the filtered training set
# model.fit(X_placeholder_Filter1.iloc[:, [0, 1, 3, 271, 563, 8, 542]], y_placeholder.astype(int))  # Use this to train model on a specific group of features
# model.fit(X_train_Filter1, y_train.astype(int))  # Use this to train model on the filtered training set
# model.fit(X_placeholder, y_placeholder.astype(int))  # Use this to train model on the entire, unfiltered training set
# model.fit(X_train, y_train.astype(int))  # Use this to train model on the entire, unfiltered training set

# Compute metrics for models scoring - Test set
predictions = model.predict(X_test_Filter1)  # Filtered training set
# predictions = model.predict(X_test_Filter1.iloc[:, [0, 1, 3, 271, 563, 8, 542]])  # Specific list of features
# predictions = model.predict(X_test)  # Entire, unfiltered training set
accuracy = metrics.accuracy_score(y_test.astype(int), predictions)
precision = metrics.precision_score(y_test.astype(int), predictions)
recall = metrics.recall_score(y_test.astype(int), predictions)
f1 = metrics.f1_score(y_test.astype(int), predictions)
kappa = metrics.cohen_kappa_score(y_test.astype(int), predictions)
cm = confusion_matrix(y_test.astype(int), predictions)
report = classification_report(y_test.astype(int), predictions)

# """
# Display metrics
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)
print('Cohen Kappa Statistic: ', kappa)
print('Confusion Matrix: ', '\n', cm, '\n')
print('Classification Report: ', '\n', report, '\n')
# """

# COMMAND ----------

# Train model on randomly generated subsets of features and group based on model scoring
populationSize = 10000
groupSize = 5  # Use this when group sizes are fixed
# numFeaturesTotal = X_train_Filter1.shape[1]
numFeaturesTotal = X_placeholder_Filter1.shape[1]  # Use this if no validation set

# Define function to create a feasible solution ("individual")
def individual(length):
    # dummy = sorted(random.sample(range(length), randint(1, numFeaturesTotal)))  # Random group size
    dummy = sorted(random.sample(range(length), groupSize))  # Fix group sizes
    return dummy

# Define a function to create a group of feasible solutions ("population")
def population(count, length):
    return [individual(length) for x in range(count)]  # "x" individuals

solutionGroup = population(populationSize, numFeaturesTotal)
highGroup = []
lowGroup = []
accuracyList = []
precisionList = []
recallList = []
f1List = []
kappaList = []

# Define boundaries for high-quality group and low-quality group
upperThreshold = 0.22201
lowerThreshold = 0.22201

# Iterate through all solutions, fit and score model, and place in either highGroup or lowGroup
for solution in solutionGroup:
  # metric = kappa  # Change this to whichever metric is to be used for scoring the model
  # model.fit(X_train_Filter1.values[:, solution], y_train.astype(int))
  model.fit(X_placeholder_Filter1.values[:, solution], y_placeholder.astype(int))
  predictions = model.predict(X_test_Filter1.values[:, solution])
  accuracy = metrics.accuracy_score(y_test.astype(int), predictions)
  accuracyList.append(accuracy)
  precision = metrics.precision_score(y_test.astype(int), predictions)
  precisionList.append(precision)
  recall = metrics.recall_score(y_test.astype(int), predictions)
  recallList.append(recall)
  f1 = metrics.f1_score(y_test.astype(int), predictions)
  f1List.append(f1)
  kappa = metrics.cohen_kappa_score(y_test.astype(int), predictions)
  kappaList.append(kappa)
  cm = confusion_matrix(y_test.astype(int), predictions)
  report = classification_report(y_test.astype(int), predictions)
  if f1 > upperThreshold:
    highGroup.append(solution)
  if f1 < lowerThreshold:
    lowGroup.append(solution)
  print(solution)

# Print number of subsets in each group. 
print("There are ", len(highGroup), " subsets producing high-quality solutions.")
print("There are ", len(lowGroup), " subsets producing low-quality solutions.")

# COMMAND ----------

import statistics
print(statistics.median(accuracyList))
print(statistics.median(precisionList))
print(statistics.median(recallList))
print(statistics.median(f1List))
print(statistics.median(kappaList))

print(statistics.mean(accuracyList))
print(statistics.mean(precisionList))
print(statistics.mean(recallList))
print(statistics.mean(f1List))
print(statistics.mean(kappaList))

# COMMAND ----------

# Define function to determine output level after defuzzification - for use below
def get_output_label(value_level, value_level_activation):
    """
    This function takes the crisp FIS output following defuzzification and outputs the label based on the
    fuzzy output function
    :param value_level: This is the defuzzified crisp output.
    :param value_level_activation: This is the y-coordinate of the output variable's density function as
    generated by the FIS inputs, rules, and crisp output.
    :return: String label
    """
    L1_activation = skfuzzy.interp_membership(support, L1, value_level)
    L2_activation = skfuzzy.interp_membership(support, L2, value_level)
    L3_activation = skfuzzy.interp_membership(support, L3, value_level)
    L4_activation = skfuzzy.interp_membership(support, L4, value_level)
    L5_activation = skfuzzy.interp_membership(support, L5, value_level)

    if value_level < 0.1:
        label = "L5"
    else:
        if value_level < 0.3:
            if (numpy.fmin(L5_activation, value_level_activation) > numpy.fmin(L4_activation,
                                                                              value_level_activation)):
                label = "L5"
            else:
                label = "L4"
        else:
            if value_level < 0.5:
                if (numpy.fmin(L4_activation, value_level_activation) > numpy.fmin(L3_activation,
                                                                                  value_level_activation)):
                    label = "L4"
                else:
                    label = "L3"
            else:
                if value_level < 0.7:
                    if (numpy.fmin(L3_activation, value_level_activation) > numpy.fmin(L2_activation,
                                                                                      value_level_activation)):
                        label = "L3"
                    else:
                        label = "L2"
                else:
                    if value_level < 0.9:
                        if (numpy.fmin(L2_activation, value_level_activation) > numpy.fmin(L1_activation,
                                                                                          value_level_activation)):
                            label = "L2"
                        else:
                            label = "L1"
                    else:
                        label = "L1"

    return label

# COMMAND ----------

# Train model on randomly generated subsets of features and group based on model scoring
populationSize = 5000  # number of subsets to generate (runs)
numReplications = 5  # number of replications of size 'populationSize'
groupSize = 5  # Use this when group sizes are fixed
# numFeaturesTotal = X_train_Filter1.shape[1]
numFeaturesTotal = X_placeholder_Filter1.shape[1]
individualRunResults = []  # List of index and FIS crisp output for each feature in a single run
consolidatedRunResults = []  # Consolidated list of all 'numReplications' run results

for run in range(numReplications):
  individualRunResults = []  # Empty the list
  # consolidatedRunResults = []  # Empty the list
  # Define function to create a feasible solution ("individual")
  def individual(length):
    random.seed()
    # dummy = sorted(random.sample(range(length), randint(1, numFeaturesTotal)))  # Random group size
    dummy = sorted(random.sample(range(length), groupSize))  # Fix group sizes
    return dummy

  # Define a function to create a group of feasible solutions ("population")
  def population(count, length):
      return [individual(length) for x in range(count)]  # "x" individuals

  solutionGroup = population(populationSize, numFeaturesTotal)
  highGroup = []
  lowGroup = []

  # Define boundaries for high-quality group and low-quality group
  upperThreshold = 0.15
  lowerThreshold = 0.15

  # Iterate through all solutions, fit and score model, and place in either highGroup or lowGroup
  for solution in solutionGroup:
    # metric = kappa  # Change this to whichever metric is to be used for scoring the model
    # model.fit(X_train_Filter1.values[:, solution], y_train.astype(int))
    model.fit(X_placeholder_Filter1.values[:, solution], y_placeholder.astype(int))
    predictions = model.predict(X_test_Filter1.values[:, solution])
    accuracy = metrics.accuracy_score(y_test.astype(int), predictions)
    precision = metrics.precision_score(y_test.astype(int), predictions)
    recall = metrics.recall_score(y_test.astype(int), predictions)
    f1 = metrics.f1_score(y_test.astype(int), predictions)
    kappa = metrics.cohen_kappa_score(y_test.astype(int), predictions)
    cm = confusion_matrix(y_test.astype(int), predictions)
    report = classification_report(y_test.astype(int), predictions)
    if f1 > upperThreshold:
      highGroup.append(solution)
    if f1 < lowerThreshold:
      lowGroup.append(solution)

  # Print number of subsets in each group. 
  print("There are ", len(highGroup), " subsets producing high-quality solutions in run ", run, ".")
  print("There are ", len(lowGroup), " subsets producing low-quality solutions in run ", run, ".")
  
  # Calculate crisp inputs for Fuzzy Inference System (FIS) based on individual features' membership in groups defined in previous cell
  maxHighFrequency = 0
  maxLowFrequency = 0
  highCountList = []
  lowCountList = []
  highNormalizedList = []
  lowNormalizedList = []
  featureIndex = []  # This is just an array of the feature indices, created for output display purposes when FIS is finished
  highGroupWeakDOM = []
  highGroupModerateDOM = []
  highGroupStrongDOM = []
  lowGroupWeakDOM = []
  lowGroupModerateDOM = []
  lowGroupStrongDOM = []
  R1 = []
  R2 = []
  R3 = []
  R4 = []
  R5 = []
  AGG = []
  defuzzifiedOutput = []
  outputDOM = []
  outputLevel = []

  # for i in range(X_train_Filter1.shape[1]):
  for i in range(X_placeholder_Filter1.shape[1]):
    featureIndex.append(i)
    highFrequency = sum(x.count(i) for x in highGroup)  # Count number of times index "i" appears in all "x" solutions appearing in highGroup
    lowFrequency = sum(x.count(i) for x in lowGroup)  # Same as highFrequency 
    highCountList.append(highFrequency)  # Add count calculated above to the corresponding index in highCountList
    highNormalizedList.append(highFrequency)  # Same - placeholder at this point; will use below
    lowCountList.append(lowFrequency)  # Same as highCountList
    lowNormalizedList.append(lowFrequency)  # Same as highNormalizedList
    if highFrequency > maxHighFrequency:
      maxHighFrequency = highFrequency  # Store highest frequency that any feature appears in highGroup
    if lowFrequency > maxLowFrequency:
      maxLowFrequency = lowFrequency  # Same as maxHighFrequency

  # Normalize by dividing each entry by the highest frequency attained by any feature    
  for i in range(len(highCountList)):
    highNormalizedList[i] = highCountList[i] / maxHighFrequency  
    lowNormalizedList[i] = lowCountList[i] / maxLowFrequency  # Could have done separate loop, but len(highCountList) = len(lowCountList) 
    # print("Index: ", featureIndex[i], ": \t High Count: ", highCountList[i], ": \t Low Count: ", lowCountList[i])
  
  # Set up FIS
  support = numpy.arange(0, 1.01, 0.01)  # Define possible crisp input values. This could be different for any fuzzy variables (same in this case)

  # Define 3-level fuzzy input membership functions
  weak = skfuzzy.trapmf(support, [0, 0, .1, .5])  # Trapezoidal membership function
  moderate = skfuzzy.trimf(support, [.1, .5, .9])  # Triangular membership function
  strong = skfuzzy.trapmf(support, [.5, .9, 1, 1])  # Trapezoidal membership function

  # Define 5-level fuzzy output membership functions
  L5 = skfuzzy.trapmf(support, [0, 0, .1, .3])  # Trapezoidal
  L4 = skfuzzy.trimf(support, [.1, .3, .5])  # Triangular
  L3 = skfuzzy.trimf(support, [.3, .5, .7])  # Triangular
  L2 = skfuzzy.trimf(support, [.5, .7, .9])  # Triangular
  L1 = skfuzzy.trapmf(support, [.7, .9, 1, 1])  # Trapezoidal

  """
  # Plot 3-level fuzzy input membership function
  fig, ax0 = plt.subplots(figsize=(5, 4))
  ax0.plot(support, weak, 'red', linewidth=1.5, label='Weak')
  ax0.plot(support, moderate, 'yellow', linewidth=1.5, label='Moderate')
  ax0.plot(support, strong, 'green', linewidth=1.5, label='Strong')

  ax0.set_title('Fuzzy Input (3-Level)')
  ax0.legend()
  plt.show(fig)
  display(fig)
  # """
  
  # Continue FIS - Fuzzification: Every feature will receive a degree of membership (DOM) in each level (Weak, Moderate, Strong) of each group (highGroup, lowGroup)  

  # for index in range(X_train_Filter1.shape[1]):
  for index in range(X_placeholder_Filter1.shape[1]):

    high_WEAK = skfuzzy.interp_membership(support, weak, highNormalizedList[index])  # Calculate DOM for WEAK level of highGroup crisp input
    highGroupWeakDOM.append(high_WEAK)
    high_MOD = skfuzzy.interp_membership(support, moderate, highNormalizedList[index])
    highGroupModerateDOM.append(high_MOD)
    high_STRONG = skfuzzy.interp_membership(support, strong, highNormalizedList[index])
    highGroupStrongDOM.append(high_STRONG)
    low_WEAK = skfuzzy.interp_membership(support, weak, lowNormalizedList[index])
    lowGroupWeakDOM.append(low_WEAK)
    low_MOD = skfuzzy.interp_membership(support, moderate, lowNormalizedList[index])
    lowGroupModerateDOM.append(low_MOD)
    low_STRONG = skfuzzy.interp_membership(support, strong, lowNormalizedList[index])
    lowGroupStrongDOM.append(low_STRONG)
    # If high-quality group membership is STRONG and low-quality group membership is WEAK then value is Level 1
    rule1 = numpy.fmin(highGroupStrongDOM[index], lowGroupWeakDOM[index])
    rule1_activation = numpy.fmin(rule1, L1)
    R1.append(rule1_activation)
    # If high-quality group membership is STRONG and low-quality group membership is NOT WEAK then value is Level 2
    rule2 = numpy.fmin(highGroupStrongDOM[index], 1 - lowGroupWeakDOM[index])
    rule2_activation = numpy.fmin(rule2, L2)
    R2.append(rule2_activation)
    # If high-quality group membership is WEAK and low-quality group membership is WEAK then value is Level 3
    rule3 = numpy.fmin(highGroupWeakDOM[index], lowGroupWeakDOM[index])
    rule3_activation = numpy.fmin(rule3, L3)
    R3.append(rule3_activation)
    # If high-quality group membership is NOT STRONG and low-quality group membership is NOT WEAK then value is Level 4
    rule4 = numpy.fmin(1 - highGroupStrongDOM[index], 1 - lowGroupWeakDOM[index])
    rule4_activation = numpy.fmin(rule4, L4)
    R4.append(rule4_activation)
    # If high-quality group membership is WEAK and low-quality group membership is STRONG then value is Level 5
    rule5 = numpy.fmin(highGroupWeakDOM[index], lowGroupStrongDOM[index])
    rule5_activation = numpy.fmin(rule5, L5)
    R5.append(rule5_activation)
    aggregated_output = numpy.fmax(R1[index], (numpy.fmax(R2[index], (numpy.fmax(R3[index], (numpy.fmax(R4[index],R5[index])))))))
    AGG.append(aggregated_output)
    value_level = skfuzzy.defuzz(support, AGG[index], 'mom')
    defuzzifiedOutput.append(value_level)
    value_level_activation = skfuzzy.interp_membership(support, AGG[index], defuzzifiedOutput[index])
    outputDOM.append(value_level_activation)
    label = get_output_label(defuzzifiedOutput[index], outputDOM[index])
    outputLevel.append(label)
    # print(index, '\t', defuzzifiedOutput[index], '\t \t', outputDOM[index], '\t', outputLevel[index])
    individualRunResults.append([index, value_level])
  sortedIndividualRunResults = sorted(individualRunResults, key=lambda x: x[1], reverse=True)  # Sort results by defuzzified crisp output
  consolidatedRunResults.append(sortedIndividualRunResults)
consolidatedRunResults = pandas.DataFrame(consolidatedRunResults) # Convert from list to Pandas DataFrame

# COMMAND ----------

print(consolidatedRunResults)

# COMMAND ----------

# Tabulate and store which features are good and how many times each feature is good.
goodFeatures = []
featureTotals = []
topFeaturesInRun = 25  # Define what constitutes as a 'Good' feature - top "N" features in a run
# Add a feature index to 'goodFeatures' if it is in the top 'topFeaturesInRun' scoring features of any of the 'numReplications' runs
for run in range(numReplications):
  for index in range(topFeaturesInRun):
    goodFeatures.append(consolidatedRunResults[index][run][0])
# Count up the number of times every features appears in 'goodFeatures' and store value in 'featureTotals'
# for index in range(X_train_Filter1.shape[1]):
for index in range(X_placeholder_Filter1.shape[1]):
  dummy = goodFeatures.count(index)
  featureTotals.append([index, dummy])
# Sort 'featureTotals' in descending order
sortedFeatureTotals = sorted(featureTotals, key=lambda x: x[1], reverse=True)

# COMMAND ----------

# Select 'best' features by virtue of their relative frequency in the top 'topFeaturesInRun' features in the 'numReplications' runs
finalFeatureSet = []
threshold = 15  # Choose the 'threshold' best features; this is a hyperparameter that may require tuning
for i in range(threshold):
  finalFeatureSet.append(sortedFeatureTotals[i][0])
  print(sortedFeatureTotals[i][0], ", ", sortedFeatureTotals[i][1])
print("Indices of final remaining features in model: ", finalFeatureSet)
# final_columns = X_train_Filter1.iloc[:, finalFeatureSet].columns.values  # Vector of names of final remaining features
final_columns = X_placeholder_Filter1.iloc[:, finalFeatureSet].columns.values  # Vector of names of final remaining features
print("Feature names for final remaining features in model: ", final_columns)

# COMMAND ----------

# Define function to iterate through all possible combinations of remaining features to obtain optimal subset of remaining features
def test_all_combinations(columns, maxFeatures, x_train, x_test, y_train, y_test):
    """
    This function loops through all possible combinations of features, trains the model, predicts model parameters,
    and outputs results to Excel
    :param location: Location of output file
    :param x_train: Pandas DataFrame for training set of predictor variables
    :param x_test: Pandas DataFrame for test set of predictor variables
    :param y_train: Response variable training set
    :param y_test: Response variable test set
    :return:
    """
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_kappa = 0
    best_accuracy_comb = (0,)
    best_precision_comb = (0,)
    best_recall_comb = (0,)
    best_f1_comb = (0,)
    best_kappa_comb = (0,)

    for feature in range(maxFeatures):
        combs = [comb for comb in itertools.combinations(range(x_train.shape[1] + 0), feature + 1)]

        for comb in combs:
            model.fit(x_train.values[:, comb], y_train)
            predictions = model.predict(x_test.values[:, comb])
            accuracy = metrics.accuracy_score(y_test, predictions)
            precision = metrics.precision_score(y_test, predictions, average='binary')
            recall = metrics.recall_score(y_test, predictions, average='binary')
            f1 = metrics.f1_score(y_test, predictions, average='binary')
            kappa = metrics.cohen_kappa_score(y_test, predictions)
            # cm = confusion_matrix(y_test, predictions)
            # report = classification_report(y_test, predictions)
            # predictions, accuracy, precision, recall, f1, kappa, cm, report = score_model(model, x_test.values[:, comb],y_test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_comb = comb
            if precision > best_precision:
                best_precision = precision
                best_precision_comb = comb
            if recall > best_recall:
                best_recall = recall
                best_recall_comb = comb
            if f1 > best_f1:
                best_f1 = f1
                best_f1_comb = comb
            if kappa > best_kappa:
                best_kappa = kappa
                best_kappa_comb = comb

            # print(comb)
            # print(feature_columns[[comb]])


    print('\n', 'Best accuracy: %f' % best_accuracy)
    print('Combination producing best accuracy: ',
          columns[[best_accuracy_comb]])  
    model.fit(x_train.values[:, best_accuracy_comb], y_train)
    predictions = model.predict(x_test.values[:, best_accuracy_comb])
    print('Confusion matrix from combination', best_accuracy_comb, ': ', '\n', confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    print('\n', 'Best precision: %f' % best_precision)
    print('Combination producing best precision: ',
          columns[[best_precision_comb]])  
    model.fit(x_train.values[:, best_precision_comb], y_train)
    predictions = model.predict(x_test.values[:, best_precision_comb])
    print('Confusion matrix from combination', best_precision_comb, ': ', '\n', confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    print('Best recall: %f' % best_recall)
    print('Combination producing best recall: ',
          columns[[best_recall_comb]])  
    model.fit(x_train.values[:, best_recall_comb], y_train)
    predictions = model.predict(x_test.values[:, best_recall_comb])
    print('Confusion matrix from combination', best_recall_comb, ': ', '\n', confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    print('Best f1: %f' % best_f1)
    print('Combination producing best f1: ',
          columns[[best_f1_comb]])  
    model.fit(x_train.values[:, best_f1_comb], y_train)
    predictions = model.predict(x_test.values[:, best_f1_comb])
    print('Confusion matrix from combination', best_f1_comb, ': ', '\n', confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    print('Best kappa: %f' % best_kappa)
    print('Combination producing best kappa: ',
          columns[[best_kappa_comb]])  
    model.fit(x_train.values[:, best_kappa_comb], y_train)
    predictions = model.predict(x_test.values[:, best_kappa_comb])
    print('Confusion matrix from combination', best_kappa_comb, ': ', '\n', confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    return

# COMMAND ----------

# Run model for final feature set
print("Indices of final remaining features in model: ", finalFeatureSet)
print("Feature names for final remaining features in model: ", final_columns)

model.fit(X_placeholder_Filter1.iloc[:, finalFeatureSet], y_placeholder.astype(int))  # Use this to train model on the filtered training set
# model.fit(X_train_Filter1.iloc[:, finalFeatureSet], y_train.astype(int))  # Use this to train model on the filtered training set

# Compute metrics for models scoring - Test set
predictions = model.predict(X_test_Filter1.iloc[:, finalFeatureSet])  # Filtered training set
accuracy = metrics.accuracy_score(y_test.astype(int), predictions)
precision = metrics.precision_score(y_test.astype(int), predictions)
recall = metrics.recall_score(y_test.astype(int), predictions)
f1 = metrics.f1_score(y_test.astype(int), predictions)
kappa = metrics.cohen_kappa_score(y_test.astype(int), predictions)
cm = confusion_matrix(y_test.astype(int), predictions)
report = classification_report(y_test.astype(int), predictions)

# """
# Display metrics
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)
print('Cohen Kappa Statistic: ', kappa)
print('Confusion Matrix: ', '\n', cm, '\n')
print('Classification Report: ', '\n', report, '\n')
# Iterate through all possible combinations of remaining features to obtain optimal subset of remaining features
maxFeatures = len(finalFeatureSet)
# test_all_combinations(final_columns, maxFeatures, X_train_Filter1.iloc[:, finalFeatureSet], X_test_Filter1.iloc[:, finalFeatureSet], y_train.astype(int), y_test.astype(int))
test_all_combinations(final_columns, maxFeatures, X_placeholder_Filter1.iloc[:, finalFeatureSet], X_test_Filter1.iloc[:, finalFeatureSet], y_placeholder.astype(int), y_test.astype(int))