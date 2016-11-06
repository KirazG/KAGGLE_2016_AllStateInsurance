#####################################################################################
#####################################################################################
##### KAGGLE COMPETITION DATASET: ALL STATE INSURANCE LOSS PREDICTION
#####################################################################################
#####################################################################################
##### FEATURE ENGINEERING: PCA FOR NUMERIC VARIABLES | loss >> LOG(loss) TRANSFORM
##### FRAMEWORK: H2O
##### ALGORITHM: GBM - GRADIANT BOOSTING MACHINE
#####################################################################################

# Cleanup the environment and load libraries
rm(list = ls(all.names = TRUE))
library(h2o)
library(dplyr)

# Initialize H2O :: DELL_LAPTOP
h2o.init(nthreads = -1, min_mem_size = "7G")

# Initialize H2O :: LENOVO_AIO
#h2o.init(nthreads = -1, min_mem_size = "3500M")

# Read Train & Test into H2O - DELL_LAPTOP
AllStateTrain.hex = h2o.uploadFile(path = "C:/02 KAGGLE/2016_AllState_Claim_Severity_Prediction/train.csv", destination_frame = "AllStateTrain.hex", header = TRUE)
AllStateTest.hex = h2o.uploadFile(path = "C:/02 KAGGLE/2016_AllState_Claim_Severity_Prediction/test.csv", destination_frame = "AllStateTest.hex", header = TRUE)

# Read Train & Test into H2O - LENOVO_AIO
#AllStateTrain.hex = h2o.uploadFile(path = "D:/10 CONTINUOUS LEARNING/83 KAGGLE/KAGGLE_COMPETITIONS/2016_01_AllState_Claim_Severity_Prediction/train.csv", destination_frame = "AllStateTrain.hex", header = TRUE)
#AllStateTest.hex = h2o.uploadFile(path = "D:/10 CONTINUOUS LEARNING/83 KAGGLE/KAGGLE_COMPETITIONS/2016_01_AllState_Claim_Severity_Prediction/test.csv", destination_frame = "AllStateTest.hex", header = TRUE)

# Transform "loss" variable to Log(loss)
AllStateTrain.hex$loss = h2o.log(AllStateTrain.hex$loss)
# No loss variable in AllStateTest.hex

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

### REMOVING UNWANTED FEATURES - BASED ON GBM FEATURE SELECTION
### FEATURE ELIMINATION: AGGRESIVE [REMOVING 61 FEATURES]

#ncol(AllStateTrain.hex)
#AllStateTrain.hex = as.h2o(select(as.data.frame(AllStateTrain.hex), -cat89, -cat104, -cat50, -cat97, -cat52, -cat10, -cat92, -cat7, -cat3, -cat65, -cat66, -cat8, -cat67, -cat42, -cat54, -cat95, -cat32, -cat19, -cat78, -cat21, -cat29, -cat51, -cat85, -cat40, -cat16, -cat41, -cat28, -cat45, -cat43, -cat98, -cat30, -cat74, -cat24, -cat88, -cat86, -cat96, -cat18, -cat59, -cat17, -cat14, -cat56, -cat71, -cat33, -cat34, -cat61, -cat46, -cat68, -cat60, -cat47, -cat63, -cat48, -cat35, -cat22, -cat55, -cat58, -cat20, -cat62, -cat70, -cat15, -cat69, -cat64))
#ncol(AllStateTrain.hex)
#ncol(AllStateTest.hex)
#AllStateTest.hex = as.h2o(select(as.data.frame(AllStateTest.hex), -cat89, -cat104, -cat50, -cat97, -cat52, -cat10, -cat92, -cat7, -cat3, -cat65, -cat66, -cat8, -cat67, -cat42, -cat54, -cat95, -cat32, -cat19, -cat78, -cat21, -cat29, -cat51, -cat85, -cat40, -cat16, -cat41, -cat28, -cat45, -cat43, -cat98, -cat30, -cat74, -cat24, -cat88, -cat86, -cat96, -cat18, -cat59, -cat17, -cat14, -cat56, -cat71, -cat33, -cat34, -cat61, -cat46, -cat68, -cat60, -cat47, -cat63, -cat48, -cat35, -cat22, -cat55, -cat58, -cat20, -cat62, -cat70, -cat15, -cat69, -cat64))
#ncol(AllStateTest.hex)

### FEATURE ELIMINATION: MODERATE [REMOVED 31 FEATURES]

ncol(AllStateTrain.hex)
AllStateTrain.hex = as.h2o(select(as.data.frame(AllStateTrain.hex), -cat30,  -cat74,  -cat24,  -cat88,  -cat86,  -cat96,  -cat18,  -cat59,  -cat17,  -cat14,  -cat56,  -cat71,  -cat33,  -cat34,  -cat61,  -cat46,  -cat68,  -cat60,  -cat47,  -cat63,  -cat48,  -cat35,  -cat22,  -cat55,  -cat58,  -cat20,  -cat62,  -cat70,  -cat15,  -cat69,  -cat64))
ncol(AllStateTrain.hex)
ncol(AllStateTest.hex)
AllStateTest.hex = as.h2o(select(as.data.frame(AllStateTest.hex), -cat30,  -cat74,  -cat24,  -cat88,  -cat86,  -cat96,  -cat18,  -cat59,  -cat17,  -cat14,  -cat56,  -cat71,  -cat33,  -cat34,  -cat61,  -cat46,  -cat68,  -cat60,  -cat47,  -cat63,  -cat48,  -cat35,  -cat22,  -cat55,  -cat58,  -cat20,  -cat62,  -cat70,  -cat15,  -cat69,  -cat64))
ncol(AllStateTest.hex)

# Split Training Data into Train & Validation set
SplitFrames = h2o.splitFrame(data = AllStateTrain.hex, ratios = 0.7, seed = 1)
ModTrain.hex = h2o.assign(data = SplitFrames[[1]], key = "ModTrain.hex")
ModTest.hex = h2o.assign(data = SplitFrames[[2]], key = "ModTest.hex")

# Attribute split for further modelling.
DepAttrib = "loss"
IndAttrib = setdiff(names(ModTrain.hex), DepAttrib)

#####################################################################################
##### IMPLEMENTING GBM ON SELECTED FEATURES                                     #####
#####################################################################################

-------------------------------------------------------------------------------------
##### GBM_01: BASED ON TOP 3 MODELS TUNED UPTO 2-NOV-2016                       #####
-------------------------------------------------------------------------------------

HyperParam = list(col_sample_rate = c(1,0.8))

gridGBM01        <- h2o.grid(algorithm = "gbm", 
                            x = IndAttrib,
                            y = DepAttrib, 
                            training_frame = ModTrain.hex, 
                            grid_id = "GRID_GBM_01",
                            hyper_params = HyperParam,
                            validation_frame = ModTest.hex, 
                            seed = 1, 
                            distribution = "gaussian",
                            # ntrees and max_depth selected based on the results from past tuning
                            ntrees = 1000,
                            max_depth = 8,
                            ## LR selected based on past tuning experience
                            learn_rate = 0.02,                                                         
                            ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                            ## learn_rate_annealing = 0.99,
                            ## Early stopping configuration
                            stopping_rounds = 3, 
                            stopping_tolerance = 1e-4,
                            stopping_metric = "deviance",
                            ## sample 80% of rows per tree
                            sample_rate = 0.8,
                            ## sample 80% of columns per split
                            ##col_sample_rate = 0.8,
                            ## score every 5 trees to make early stopping reproducible
                            score_tree_interval = 5)

# Get grid:
gridGBM = h2o.getGrid(grid_id = "GRID_GBM_01", sort_by = "mse")

GBMModel = h2o.getModel(model_id = gridGBM@model_ids[[1]])
predGBM = h2o.predict(object = GBMModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_05112016_01.csv", row.names = FALSE)

GBMModel = h2o.getModel(model_id = gridGBM@model_ids[[2]])
predGBM = h2o.predict(object = GBMModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_05112016_02.csv", row.names = FALSE)


-------------------------------------------------------------------------------------
##### GBM_02: BASED ON TOP 3 MODELS TUNED UPTO 2-NOV-2016                       #####
-------------------------------------------------------------------------------------

HyperParam = list(col_sample_rate = c(0.4,0.6,0.8,1),
                  max_depth = c(6,7,8),
                  learn_rate = c(0.02,0.03),
                  learn_rate_annealing = c(1,0.999))

gridGBM02        <- h2o.grid(algorithm = "gbm", 
                             x = IndAttrib,
                             y = DepAttrib, 
                             training_frame = ModTrain.hex, 
                             grid_id = "GRID_GBM_02",
                             hyper_params = HyperParam,
                             validation_frame = ModTest.hex, 
                             seed = 1, 
                             distribution = "gaussian",
                             # ntrees and max_depth selected based on the results from past tuning
                             ntrees = 1500,
                             #max_depth = 8,
                             ## LR selected based on past tuning experience
                             #learn_rate = 0.02,                                                         
                             ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                             ## learn_rate_annealing = 0.99,
                             ## Early stopping configuration
                             stopping_rounds = 3, 
                             stopping_tolerance = 1e-4,
                             stopping_metric = "deviance",
                             ## sample 80% of rows per tree
                             sample_rate = 0.8,
                             ## sample 80% of columns per split
                             ##col_sample_rate = 0.8,
                             ## score every 5 trees to make early stopping reproducible
                             score_tree_interval = 5)


-------------------------------------------------------------------------------------
##### GBM_03: COLUMN SAMPLE RATE TUNING BASED ON LAST RUN                       #####
-------------------------------------------------------------------------------------

# Hyperparameters for Round 1:
#HyperParam = list(col_sample_rate = c(0.33,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.90,0.95,1), max_depth = c(6), learn_rate = c(0.02), learn_rate_annealing = c(1))
    
### OBSERVATION: 
#   75% WEIGHT OF ACCURACY AND 25% WEIGHT OF MAE_SPREAD >> 0.50, 0.33, 0.65 ARE TOP3 COL_SAMP_RATE
#   50% WEIGHT OF ACCURACY AND 50% WEIGHT OF MAE_SPREAD >> 0.50, 0.95, 0.65 ARE BEST COL_SAMPLE_RATE
    

# Hyperparameters for Round 2:
HyperParam = list(col_sample_rate = c(0.25, 0.35, 0.5),
                  max_depth = c(6),
                  learn_rate = c(0.02),
                  learn_rate_annealing = c(1))

gridGBM03        <- h2o.grid(algorithm = "gbm", 
                             x = IndAttrib,
                             y = DepAttrib, 
                             training_frame = ModTrain.hex, 
                             grid_id = "GRID_GBM_04",
                             hyper_params = HyperParam,
                             validation_frame = ModTest.hex, 
                             seed = 1, 
                             distribution = "gaussian",
                             # ntrees and max_depth selected based on the results from past tuning
                             ntrees = 1500,
                             #max_depth = 8,
                             ## LR selected based on past tuning experience
                             #learn_rate = 0.02,                                                         
                             ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                             ## learn_rate_annealing = 0.99,
                             ## Early stopping configuration
                             stopping_rounds = 3, 
                             stopping_tolerance = 1e-4,
                             stopping_metric = "deviance",
                             ## sample 80% of rows per tree
                             sample_rate = 1,
                             ## sample 80% of columns per split
                             ##col_sample_rate = 0.8,
                             ## score every 5 trees to make early stopping reproducible
                             score_tree_interval = 5)


# Generating Predicting For Top 3 Models:
# #3: GRID_GBM_02_model_8
# #2: GRID_GBM_04_model_1
# #1: GRID_GBM_04_model_0

# FOR #3: GRID_GBM_02_model_8
GBMModel = h2o.getModel(model_id = "GRID_GBM_02_model_8")
predGBM = h2o.predict(object = GBMModel, newdata = AllStateTest.hex)
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_05112016_01.csv", row.names = FALSE)

# FOR #3: GRID_GBM_04_model_1
GBMModel = h2o.getModel(model_id = "GRID_GBM_04_model_1")
predGBM = h2o.predict(object = GBMModel, newdata = AllStateTest.hex)
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_05112016_02.csv", row.names = FALSE)

# FOR #3: GRID_GBM_04_model_0
GBMModel = h2o.getModel(model_id = "GRID_GBM_04_model_0")
predGBM = h2o.predict(object = GBMModel, newdata = AllStateTest.hex)
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_05112016_03.csv", row.names = FALSE)

### KGALLE BEST SCORE SO FAR: 1145.44359 | VAL_MAE: 0.421122


# Hyperparameters for Round 2:
HyperParam = list(col_sample_rate = c(0.25, 0.35, 0.5), sample_rate = c(0.4,0.5,0.6,0.7,0.8,0.9,1))

gridGBM          <- h2o.grid(algorithm = "gbm", 
                             x = IndAttrib,
                             y = DepAttrib, 
                             training_frame = ModTrain.hex, 
                             grid_id = "GRID_GBM_05",
                             hyper_params = HyperParam,
                             validation_frame = ModTest.hex, 
                             seed = 1, 
                             distribution = "gaussian",
                             # ntrees and max_depth selected based on the results from past tuning
                             ntrees = 1500,
                             max_depth = 6,
                             learn_rate = 0.02,                                                         
                             ## learn_rate_annealing = 0.99,
                             stopping_rounds = 3, 
                             stopping_tolerance = 1e-4,
                             stopping_metric = "deviance",
                             #sample_rate = 1,
                             ## col_sample_rate = 0.8,
                             ## score every 5 trees to make early stopping reproducible
                             score_tree_interval = 5)