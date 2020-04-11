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

# Load data from individual .csv files in FileStore as Spark DataFrames
inputSparkDF_337123_1T_APR = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/union_337123_1T_Apr-7b0c0.csv')

# COMMAND ----------

#this section will convert Spark dataframe to python pandas data frame which is used in tsfresh alorithmn
inputData = inputSparkDF_337123_1T_APR.toPandas()

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
#print(inputData['aoi_defect'].sum())
#print(inputData['ict_defect'].sum())
#print(inputData['fa_defect'].sum())
#print(inputData['any_defect'].sum())

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

#creating a reference column to find which panel it is.
inputData["BarcodeSilkscreenRefID"] = inputData['barcd'].map(str) + "_" + inputData['silkscrn_nbr'].map(str) + "_" + inputData['ref_id']

# COMMAND ----------

# Sort by Barcode_Silkscreen_Reference Designator, then Component Pin Number, then presence of any defect
inputData = inputData.sort_values(['BarcodeSilkscreenRefID', 'cmpntpin_nbr', 'ict_defect'])

# COMMAND ----------

# Create a DataFrame for numeric input data to model as time-series-like within the barcode-silkscreen-reference_id index
#from the data set columns we only choose the below columns as features and tsfresh will create more new features using these columns
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

#get rid of duplicates column
numericInputData_red1 = numericInputData.drop_duplicates(keep='last')
numericInputData_red1b = numericInputData.drop_duplicates(subset=['id', 'pin'], keep='last')

# COMMAND ----------

# Create a DataFrame for each response variable
# finalzing only the columns used for model training
responseICT = pandas.DataFrame()
responseICT['id'] = numericInputData_red1b['id']
responseICT['ict_defect'] = numericInputData_red1b['ict_defect']

# COMMAND ----------

#sort and remove duplicates
responseICT = responseICT.sort_values(['id', 'ict_defect'])
responseICT_red1 = responseICT.drop_duplicates(keep='last')

# COMMAND ----------

responseICT_red2 = responseICT_red1[~((responseICT_red1.duplicated(subset=['id'], keep=False)) & (responseICT_red1['ict_defect'] == 0))]

# COMMAND ----------

# Response variable must be in the form of a Pandas.Series in order for TSFRESH to work
#responseAOI_tsfresh = pandas.Series(responseAOI_red2.values[:, 1], index=responseAOI_red2.id)
responseICT_tsfresh = pandas.Series(responseICT_red2.values[:, 1], index=responseICT_red2.id)

# COMMAND ----------

# Generate features from "TS" data using TSFRESH package

extraction_settings=ComprehensiveFCParameters()  # Generate all possible features
# extraction_settings=EfficientFCParameters()  # Generate only features with "efficient" processing time
# extraction_settings=MinimalFCParameters()  # Generate minimal, quickly-generated features

X = extract_features(numericInputData_red1b.iloc[:, 0:7], column_id='id', column_sort='pin', default_fc_parameters=extraction_settings, impute_function=impute)

# COMMAND ----------

# Split data into training set and test set
X_placeholder, X_test, y_placeholder, y_test = model_selection.train_test_split(X, responseICT_tsfresh, test_size = 0.2, random_state=2, shuffle=True)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_placeholder, y_placeholder, test_size = 0.25, random_state=2, shuffle=True)

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

final_columns = ['sp_v_p__quantile__q_0.7' 'sp_h__quantile__q_0.1' 'sp_v_p__c3__lag_1'
 'sp_h__abs_energy' 'sp_h__sum_values' 'sp_h__mean'
 'sp_o_x__longest_strike_above_mean' 'sp_v_p__abs_energy'
 'sp_v_p__quantile__q_0.6' 'sp_h__fft_coefficient__coeff_0__attr_"real"'
 'sp_h__quantile__q_0.7' 'sp_h__c3__lag_1' 'sp_h__quantile__q_0.4'
 'sp_h__quantile__q_0.8' 'sp_h__fft_coefficient__coeff_45__attr_"abs"']

# COMMAND ----------

#get only the necessary features for predicting.
finalFeatureSet = []
finalFeatureSet = X_test_Filter1[final_columns]

# COMMAND ----------

#save the model as pickle file
filename = ‘modelfile_20192201.pkl’
pickle.dump(clf, open(filename, ‘wb’), protocol=2)

# COMMAND ----------

#load model here from pickle 
#convert the variables into dataframe.
#testData = pd.DataFrame({‘Age’: Age, ‘KM’: KM}, index=[0])
#X = testData[[“Age”, “KM”]].values

X = inputSparkDF.topandas()
y=np.array([0])
from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
# load the model from disk
filename = ‘modelfile_20192201.pkl’
loaded_model = pickle.load(open(filename, ‘rb’))
result = loaded_model.score(X_placeholder_Filter1.iloc[:, finalFeatureSet], y)

# COMMAND ----------

print(str(result))

# COMMAND ----------

#result is what should be return out as model output and business logic should be applied to take actions like sending it to uptime system or alerting etc.
#the inference code stops here.

# COMMAND ----------

#below here is a sample training model code with only the features selected above.

# COMMAND ----------

#below is only for reference.

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

# COMMAND ----------

#save the model as pickle file
filename = ‘modelfile_20192201.pkl’
pickle.dump(model, open(filename, ‘wb’), protocol=2)

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

    # Encode categorical variables - replace category levels with numeric values
    encoder = category_encoders.OrdinalEncoder()  # Not *best* but pretty close and obtained quickly
    # encoder = category_encoders.LeaveOneOutEncoder()  # Similar to target encoding; best overall result
    # encoder = category_encoders.TargetEncoder()  
    # encoder = category_encoders.HashingEncoder() 
    # encoder = category_encoders.BinaryEncoder() 
    X_train_E = encoder.fit_transform(X_train, Y_train)  # Fit the training set with the encodings
    X_test_E = encoder.transform(X_test)  # Convert categories in test set using the encoding previously fitted

# COMMAND ----------

# Filter #1 - Reduce Features Using TSFRESH
reduced_feature_set = select_features(X_train_E, Y_train)
remaining_column_names = reduced_feature_set.columns.values  # Names of remaining columns

# Relabel X_train for use below
X_train_Filter1 = reduced_feature_set

# Ensure that test set columns correspond to training set columns
X_test_Filter1 = X_test_E[remaining_column_names]

# COMMAND ----------

#pull only the columns that is finalized as most important features that help the prediction. Output from skfuzzy logic from philips feature extraction
#form the data set only with those variable
inputDF = pandas.DataFrame()
inputDF['Barcode'] = X_test_Filter1['barcd']
inputDF['SPI Test ID'] = X_test_Filter1['spi_test_id']
inputDF['Silkscreen'] = X_test_Filter1['silkscrn_nbr']
inputDF['Reference ID'] = X_test_Filter1['ref_id']
inputDF['Component Pin Number'] = X_test_Filter1['cmpntpin_nbr']
inputDF['Component Part Number'] = X_test_Filter1['cmpnt_part_num']
#inputDF['Prediction'] = model.predict(X_test_Filter1)
#inputDF['Actual'] = inputDataTest['ict_defect']

inputSparkDF = spark.createDataFrame(inputDF)

# COMMAND ----------

#save the model as pickle file
filename = ‘modelfile_20192201.pkl’
pickle.dump(clf, open(filename, ‘wb’), protocol=2)

# COMMAND ----------

#load model here from pickle 
#convert the variables into dataframe.
#testData = pd.DataFrame({‘Age’: Age, ‘KM’: KM}, index=[0])
#X = testData[[“Age”, “KM”]].values

X = inputSparkDF.topandas()
y=np.array([0])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# load the model from disk
filename = ‘modelfile_20192201.pkl’
loaded_model = pickle.load(open(filename, ‘rb’))
result = loaded_model.score(X_scaled, y)

# COMMAND ----------

print(str(result))