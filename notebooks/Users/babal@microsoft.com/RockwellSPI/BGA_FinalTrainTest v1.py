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
import scipy

# Import specialized packages within libraries
from sklearn import ensemble, metrics, model_selection, neighbors, preprocessing, svm, tree, naive_bayes
from sklearn.metrics import classification_report, confusion_matrix
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from random import randint
import matplotlib.pylab as plt

# Set random seed
seed = 7
numpy.random.seed(seed)

# COMMAND ----------

# Load data from individual .csv files in FileStore as Spark DataFrames
inputTrain_114336_2T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_114336_2T-a8d65.csv')
inputTest_114336_2T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/114336_2T_TestOctober2018-12c60.csv')
# inputSparkDF_200188_4T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_200188_4T-3bbd1.csv')
# inputSparkDF_200189_4T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_200189_4T_Part1-1743d.csv')
# inputSparkDF_200189_4T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_200189_4T_Part2-d931b.csv')
# inputSparkDF_221364_1T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_221364_1T_Part1-4337f.csv')
# inputSparkDF_221364_1T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_221364_1T_Part2-34c2e.csv')
# inputSparkDF_264469_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_264469_1T-0f7f5.csv')
# inputSparkDF_302139_2T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_302139_2T-f7dc2.csv')
# inputSparkDF_337123_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_337123_1T-c1789.csv')
# inputSparkDF_339721_1T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_339721_1T_Part1-5d7d5.csv')
# inputSparkDF_339721_1T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_339721_1T_Part2-da85a.csv')
# inputSparkDF_339723_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_339723_1T-46362.csv')
# inputSparkDF_351299_1T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_351299_1T_Part1-fd4eb.csv')
# inputSparkDF_351299_1T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_351299_1T_Part2-e854a.csv')
# inputSparkDF_358387_2T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_358387_2T-2390b.csv')
# inputSparkDF_375220_2T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_375220_2T-19efe.csv')
# inputSparkDF_375459_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_375459_1T-e2ab9.csv')
#inputSparkDF_386238_1T_Part1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_386238_1T_Part1-5bb86.csv')
# inputSparkDF_386238_1T_Part2 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_386238_1T_Part2-5abff.csv')
# inputSparkDF_420084_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_420084_1T-8432d.csv')
# inputSparkDF_454408_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_454408_1T-c5fe8.csv')
# inputSparkDF_474138_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_474138_1T-6a500.csv')
# inputSparkDF_478000_1T = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/InputData_478000_1T-d7f61.csv')

# COMMAND ----------

print((inputTrain_114336_2T.count(), len(inputTrain_114336_2T.columns)))
print((inputTest_114336_2T.count(), len(inputTest_114336_2T.columns)))

# COMMAND ----------

# merged = inputSparkDF_351299_1T_Part1.unionAll(inputSparkDF_351299_1T_Part2)
# inputData_351299_1T = merged.toPandas()
inputTrain = inputTrain_114336_2T.toPandas()
inputTest = inputTest_114336_2T.toPandas()

# COMMAND ----------

inputTrain.duplicated(subset=['barcd', 'spi_test_id', 'ref_id', 'silkscrn_nbr', 'cmpntpin_nbr', 'cmpntpin_size_x_axis', 'cmpntpin_size_y_axis', 'solder_paste_offst_x_axis', 'solder_paste_offst_y_axis', 'pad_stncl_hgt', 'pad_spi_result', 'pad_spi_defct_type', 'solder_paste_vol_pct', 'solder_paste_hgt', 'solder_paste_area_pct', 'cmpnt_part_num', 'stncl_surf_area_ratio'], keep='first').sum()

# COMMAND ----------

inputTrain = inputTrain[~((inputTrain.duplicated(subset=['barcd', 'spi_test_id', 'ref_id', 'silkscrn_nbr', 'cmpntpin_nbr', 'cmpntpin_size_x_axis', 'cmpntpin_size_y_axis', 'solder_paste_offst_x_axis', 'solder_paste_offst_y_axis', 'pad_stncl_hgt', 'pad_spi_result', 'pad_spi_defct_type', 'solder_paste_vol_pct', 'solder_paste_hgt', 'solder_paste_area_pct', 'cmpnt_part_num', 'stncl_surf_area_ratio'], keep=False)) & (inputTrain['ict_defect'] == 0))]

# COMMAND ----------

print(inputTrain.shape)

# COMMAND ----------

inputTest.duplicated(subset=['barcd', 'spi_test_id', 'ref_id', 'silkscrn_nbr', 'cmpntpin_nbr', 'cmpntpin_size_x_axis', 'cmpntpin_size_y_axis', 'solder_paste_offst_x_axis', 'solder_paste_offst_y_axis', 'pad_stncl_hgt', 'pad_spi_result', 'pad_spi_defct_type', 'solder_paste_vol_pct', 'solder_paste_hgt', 'solder_paste_area_pct', 'cmpnt_part_num', 'stncl_surf_area_ratio'], keep='first').sum()

# COMMAND ----------

inputTest = inputTest[~((inputTest.duplicated(subset=['barcd', 'spi_test_id', 'ref_id', 'silkscrn_nbr', 'cmpntpin_nbr', 'cmpntpin_size_x_axis', 'cmpntpin_size_y_axis', 'solder_paste_offst_x_axis', 'solder_paste_offst_y_axis', 'pad_stncl_hgt', 'pad_spi_result', 'pad_spi_defct_type', 'solder_paste_vol_pct', 'solder_paste_hgt', 'solder_paste_area_pct', 'cmpnt_part_num', 'stncl_surf_area_ratio'], keep=False)) & (inputTest['ict_defect'] == 0))]

# COMMAND ----------

print(inputTest.shape)

# COMMAND ----------

# Select columns from Pandas DataFrame to serve as features (Predictors) and column to serve as label (Response)
initial_feature_set_names = inputTrain.columns.values[0:17]  # Designate leftmost columns for features
initial_feature_set_names = numpy.delete(initial_feature_set_names, [0, 1])  # Remove columns we don't want
print(initial_feature_set_names)
label_name = inputTrain.columns.values[inputTrain.shape[1] - 3]  # Designate column for response variable
print(label_name)

# Split input DataFrame into separate DataFrames for features and label
X_train = inputTrain[initial_feature_set_names]
X_test = inputTest[initial_feature_set_names]
Y_train = inputTrain[label_name]
Y_test = inputTest[label_name]


# COMMAND ----------

print(X_train.shape, '\n', X_test.shape)

# COMMAND ----------

    # Encode categorical variables - replace category levels with numeric values
    encoder = category_encoders.OrdinalEncoder()  # Not *best* but pretty close and obtained quickly
    # encoder = category_encoders.LeaveOneOutEncoder()  # Similar to target encoding; best overall result
    # encoder = category_encoders.TargetEncoder()  
    # encoder = category_encoders.HashingEncoder() 
    # encoder = category_encoders.BinaryEncoder() 
    X_train_E = encoder.fit_transform(X_train, Y_train)  # Fit the training set with the encodings
    X_test_E = encoder.transform(X_test)  # Convert categories in test set using the encoding previously fitted

# COMMAND ----------

print(Y_train.sum())
print(Y_test.sum())

# COMMAND ----------

# Filter #1 - Reduce Features Using TSFRESH
reduced_feature_set = select_features(X_train_E, Y_train)
remaining_column_names = reduced_feature_set.columns.values  # Names of remaining columns

# Relabel X_train for use below
X_train_Filter1 = reduced_feature_set

# Ensure that test set columns correspond to training set columns
X_test_Filter1 = X_test_E[remaining_column_names]

# COMMAND ----------

print("Number of features remaining in model: ", '\t', X_train_Filter1.shape[1])
print("Column names of features remaining in model: ", '\n', remaining_column_names, '\n')

# COMMAND ----------

# Train model - Select only one of the options below
# model = tree.DecisionTreeClassifier() # Decision Tree
model = ensemble.RandomForestClassifier(random_state=seed, n_estimators=20)  # Random Forest
# model = neighbors.KNeighborsClassifier()  # K-Nearest Neighbor
# model = svm.SVC()  # Support Vector Machines
# model = naive_bayes.GaussianNB()  # Naive Bayes

# Fit model 
# model.fit(X_train_E, Y_train)  # Use this to train model on the entire training set
model.fit(X_train_Filter1, Y_train)  # Use this to train model on the entire training set

# Compute metrics for models scoring
# predictions = model.predict(X_test_E)
predictions = model.predict(X_test_Filter1)
accuracy = metrics.accuracy_score(Y_test, predictions)
precision = metrics.precision_score(Y_test, predictions)
recall = metrics.recall_score(Y_test, predictions)
f1 = metrics.f1_score(Y_test, predictions)
kappa = metrics.cohen_kappa_score(Y_test, predictions)
cm = confusion_matrix(Y_test, predictions)
report = classification_report(Y_test, predictions)

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

outputDF = pandas.DataFrame()
outputDF['Barcode'] = inputDataTest['barcd']
outputDF['SPI Test ID'] = inputDataTest['spi_test_id']
outputDF['Silkscreen'] = inputDataTest['silkscrn_nbr']
outputDF['Reference ID'] = inputDataTest['ref_id']
outputDF['Component Pin Number'] = inputDataTest['cmpntpin_nbr']
outputDF['Component Part Number'] = inputDataTest['cmpnt_part_num']
outputDF['Prediction'] = model.predict(X_test_Filter1)
outputDF['Actual'] = inputDataTest['ict_defect']

outputSparkDF = spark.createDataFrame(outputDF)



# COMMAND ----------

display(outputSparkDF)

# COMMAND ----------

# Train model on randomly generated subsets of features and group based on model scoring
populationSize = 100
groupSize = 5  # Use this when group sizes are fixed
numFeaturesTotal = X_train_Filter1.shape[1]

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
upperThreshold = 0.1
lowerThreshold = 0.1

# Iterate through all solutions, fit and score model, and place in either highGroup or lowGroup
for solution in solutionGroup:
  # metric = kappa  # Change this to whichever metric is to be used for scoring the model
  model.fit(X_train_Filter1.values[:, solution], Y_train)
  predictions = model.predict(X_test_Filter1.values[:, solution])
  accuracy = metrics.accuracy_score(Y_test, predictions)
  accuracyList.append(accuracy)
  precision = metrics.precision_score(Y_test, predictions)
  precisionList.append(precision)
  recall = metrics.recall_score(Y_test, predictions)
  recallList.append(recall)
  f1 = metrics.f1_score(Y_test, predictions)
  f1List.append(f1)
  kappa = metrics.cohen_kappa_score(Y_test, predictions)
  kappaList.append(kappa)
  cm = confusion_matrix(Y_test, predictions)
  report = classification_report(Y_test, predictions)
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
populationSize = 75  # number of subsets to generate (runs)
numReplications = 5  # number of replications of size 'populationSize'
groupSize = 5  # Use this when group sizes are fixed
numFeaturesTotal = X_train_Filter1.shape[1]
individualRunResults = []  # List of index and FIS crisp output for each feature in a single run
consolidatedRunResults = []  # Consolidated list of all 'numReplications' run results

for run in range(numReplications):
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
  upperThreshold = 0.321
  lowerThreshold = 0.321

  # Iterate through all solutions, fit and score model, and place in either highGroup or lowGroup
  for solution in solutionGroup:
    # metric = kappa  # Change this to whichever metric is to be used for scoring the model
    model.fit(X_train_Filter1.values[:, solution], Y_train)
    predictions = model.predict(X_test_Filter1.values[:, solution])
    accuracy = metrics.accuracy_score(Y_test, predictions)
    precision = metrics.precision_score(Y_test, predictions)
    recall = metrics.recall_score(Y_test, predictions)
    f1 = metrics.f1_score(Y_test, predictions)
    kappa = metrics.cohen_kappa_score(Y_test, predictions)
    cm = confusion_matrix(Y_test, predictions)
    report = classification_report(Y_test, predictions)
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

  for i in range(X_train.shape[1]):
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

  for index in range(X_train.shape[1]):
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
    value_level = skfuzzy.defuzz(support, AGG[index], 'centroid')
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

# Tabulate and store which features are good and how many times each feature is good.
goodFeatures = []
featureTotals = []
topFeaturesInRun = 5  # Define what constitutes as a 'Good' feature - top "N" features in a run
# Add a feature index to 'goodFeatures' if it is in the top 'topFeaturesInRun' scoring features of any of the 'numReplications' runs
for run in range(numReplications):
  for index in range(topFeaturesInRun):
    goodFeatures.append(consolidatedRunResults[index][run][0])
# Count up the number of times every features appears in 'goodFeatures' and store value in 'featureTotals'
for index in range(X_train_Filter1.shape[1]):
  dummy = goodFeatures.count(index)
  featureTotals.append([index, dummy])
# Sort 'featureTotals' in descending order
sortedFeatureTotals = sorted(featureTotals, key=lambda x: x[1], reverse=True)

# COMMAND ----------

# Select 'best' features by virtue of their relative frequency in the top 'topFeaturesInRun' features in the 'numReplications' runs
finalFeatureSet = []
threshold = 5  # Choose the 'threshold' best features; this is a hyperparameter that may require tuning
for i in range(threshold):
  finalFeatureSet.append(sortedFeatureTotals[i][0])
print("Indices of final remaining features in model: ", finalFeatureSet)
final_columns = X_train_Filter1.iloc[:, finalFeatureSet].columns.values  # Vector of names of final remaining features
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

# Iterate through all possible combinations of remaining features to obtain optimal subset of remaining features
# maxFeatures = len(finalFeatureSet)
# test_all_combinations(final_columns, maxFeatures, X_train_Filter1.iloc[:, finalFeatureSet], X_test_Filter1.iloc[:, finalFeatureSet], Y_train, Y_test)

maxFeatures = len(X_train_Filter1)
test_all_combinations(remaining_column_names, maxFeatures, X_train_Filter1, X_test_Filter1, Y_train, Y_test)