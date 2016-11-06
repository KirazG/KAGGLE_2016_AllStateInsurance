#####################################################################################
#####################################################################################
##### KAGGLE COMPETITION DATASET: ALL STATE INSURANCE LOSS PREDICTION
#####################################################################################
#####################################################################################
##### FEATURE ENGINEERING: PCA FOR NUMERIC VARIABLES | loss >> LOG(loss) TRANSFORM
##### FRAMEWORK: H2O
##### ALGORITHM: XGB IN R - EXTREME GRADIANT BOOSTING IN R
#####################################################################################

# Cleanup the environment and load libraries
rm(list = ls(all.names = TRUE))
library(h2o)
library(dplyr)
library(xgboost)
library(dummies)

# Initialize H2O :: DELL_LAPTOP
h2o.init(nthreads = -1, min_mem_size = "5G")

# Initialize H2O :: LENOVO_AIO
#h2o.init(nthreads = -1, min_mem_size = "3500M")

# Read Train & Test into H2O - DELL_LAPTOP
AllStateTrain.hex = h2o.uploadFile(path = "C:/02 KAGGLE/2016_AllState_Claim_Severity_Prediction/train.csv", destination_frame = "AllStateTrain.hex", header = TRUE)
AllStateTest.hex = h2o.uploadFile(path = "C:/02 KAGGLE/2016_AllState_Claim_Severity_Prediction/test.csv", destination_frame = "AllStateTest.hex", header = TRUE)

# Read Train & Test into H2O - LENOVO_AIO
#AllStateTrain.hex = h2o.uploadFile(path = "D:/10 CONTINUOUS LEARNING/83 KAGGLE/KAGGLE_COMPETITIONS/2016_01_AllState_Claim_Severity_Prediction/train.csv", destination_frame = "AllStateTrain.hex", header = TRUE)
#AllStateTest.hex = h2o.uploadFile(path = "D:/10 CONTINUOUS LEARNING/83 KAGGLE/KAGGLE_COMPETITIONS/2016_01_AllState_Claim_Severity_Prediction/test.csv", destination_frame = "AllStateTest.hex", header = TRUE)

# Save and remove id column from Train & Test dataset
TrainId = AllStateTrain.hex$id
TestId = AllStateTest.hex$id
AllStateTrain.hex$id = NULL
AllStateTest.hex$id = NULL

# Variable names
vFactors = paste0("cat", 1:116)
vNumbers = paste0("cont", 1:14)

# Perform PCA on numeric data >> Remove original attributes >> Select & attach important components
PCA = h2o.prcomp(training_frame = AllStateTrain.hex, x = vNumbers, k = length(vNumbers), model_id = "PCA_01", transform = "STANDARDIZE", seed = 1)

# Remove Continuous variable and add PCA Score - Train data frame
AllStateTrainPCA.hex = h2o.assign(data = h2o.predict(object = PCA, newdata = AllStateTrain.hex), key = "AllStateTrainPCA.hex")
AllStateTrain.hex[ ,vNumbers] = NULL
AllStateTrain.hex = h2o.cbind(AllStateTrain.hex, AllStateTrainPCA.hex[ ,1:12])

# Remove Continuous variable and add PCA Score - Test Data Frame
AllStateTestPCA.hex = h2o.assign(data = h2o.predict(object = PCA, newdata = AllStateTest.hex), key = "AllStateTestPCA.hex")
AllStateTest.hex[ ,vNumbers] = NULL
AllStateTest.hex = h2o.cbind(AllStateTest.hex, AllStateTestPCA.hex[ ,1:12])

dfAllStateTrain = as.data.frame(x = AllStateTrain.hex)
dfAllStateTest = as.data.frame(x = AllStateTest.hex)
# Shutdown H2O after PCA is done !
h2o.shutdown(prompt = FALSE)

# Transform "loss" variable to Log(loss)
dfAllStateTrain$loss = log(dfAllStateTrain$loss)
# No loss variable in AllStateTest.hex

# Convert Train and Test data frames into numerical
dfAllStateTrain = dummy.data.frame(data = dfAllStateTrain, names = names(dfAllStateTrain)[1:116])
dfAllStateTest = dummy.data.frame(data = dfAllStateTest, names = names(dfAllStateTest)[1:116])


# Split Training Data into Train & Validation set
set.seed(1)
SRS = sample(x = 1:nrow(dfAllStateTrain), size = 0.7*nrow(dfAllStateTrain))
dfModTrain = dfAllStateTrain[SRS, ]
dfModTest = dfAllStateTrain[-SRS, ]

# Attribute split for further modelling.
DepAttrib = "loss"
IndAttrib = setdiff(names(dfModTrain), DepAttrib)

#####################################################################################
##### IMPLEMENTING XGB ON ALL FEATURES                                          #####
#####################################################################################


xgbParams = list(objective = "reg:linear", eta = 0.3, max.depth = 5, subsample = 0.5, colsample_bytree = 0.5)

xgbModel001 = xgboost(data = data.matrix(frame = dfModTrain), label = "loss", params = xbgParams, nrounds = 1, verbose = 2, early.stop.round = 3,maximize = TRUE )
