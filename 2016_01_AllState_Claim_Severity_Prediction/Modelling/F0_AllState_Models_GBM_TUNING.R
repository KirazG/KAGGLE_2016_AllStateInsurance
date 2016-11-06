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

# Split Training Data into Train & Validation set
SplitFrames = h2o.splitFrame(data = AllStateTrain.hex, ratios = 0.7, seed = 1)
ModTrain.hex = h2o.assign(data = SplitFrames[[1]], key = "ModTrain.hex")
ModTest.hex = h2o.assign(data = SplitFrames[[2]], key = "ModTest.hex")

# Attribute split for further modelling.
DepAttrib = "loss"
IndAttrib = setdiff(names(ModTrain.hex), DepAttrib)

#####################################################################################
##### GBM TUNING IMPLEMENTATION                                                 #####
#####################################################################################

-------------------------------------------------------------------------------------
##### GBM: TREE DEPTH TUNING                                                    #####
-------------------------------------------------------------------------------------

HyperParam = list(max_depth = seq(1,29,2))

gridDepthSearch <- h2o.grid(algorithm = "gbm", 
                            x = IndAttrib,
                            y = DepAttrib, 
                            training_frame = ModTrain.hex, 
                            grid_id = "GRID_DEPTH",
                            hyper_params = HyperParam,
                            validation_frame = ModTest.hex, 
                            seed = 1, 
                            distribution = "gaussian",
                            ntrees = 7500,
                            ## smaller learning rate is better
                            ## Due to learning_rate_annealing, we can start with a bigger learning rate
                            learn_rate = 0.05,                                                         
                            ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                            learn_rate_annealing = 0.99,
                            ## Early stopping configuration
                            stopping_rounds = 3, 
                            stopping_tolerance = 1e-4,
                            stopping_metric = "MSE",
                            ## sample 80% of rows per tree
                            sample_rate = 0.8,
                            ## sample 80% of columns per split
                            col_sample_rate = 0.8,
                            ## score every 10 trees to make early stopping reproducible
                            score_tree_interval = 10)

# Get grid:
gridDepth = h2o.getGrid(grid_id = "GRID_DEPTH", sort_by = "mse")

# Check error metrics for the models
lapply(gridDepth@model_ids, function(x) {h2o.mae(object = h2o.getModel(x), train = TRUE, valid = TRUE)})
lapply(gridDepth@model_ids, function(x) {h2o.rmse(object = h2o.getModel(x), train = TRUE, valid = TRUE)})

# Check the spread between Valid and Train metrics: MAE and RMSE
lapply(lapply(gridDepth@model_ids, function(x) {h2o.mae(object = h2o.getModel(x), train = TRUE, valid = TRUE)}), function(y){y["valid"] - y["train"]})

lapply(lapply(gridDepth@model_ids, function(x) {h2o.rmse(object = h2o.getModel(x), train = TRUE, valid = TRUE)}), function(y){y["valid"] - y["train"]})

### MAX_DEPTH TUNING OUTCOME:
###     Based on Top 5 models - max_depth range = 9-17
###     Based on Top 3 models - max_depth range = 11-15
###     Based on Observation  - max_depth range = 9-15

TopModel = h2o.getModel(model_id = gridDepth@model_ids[[1]])
TopModel = h2o.getModel(model_id = "GRID_DEPTH_model_4")

# Save the features from all 10 models
for(i in 1:10)
{
    ImpVar = h2o.varimp(h2o.getModel(model_id = gridDepth@model_ids[[i]]))
    write.csv(x = as.data.frame(x = ImpVar), file = paste0("FEATURES_GBM_GRID_MaxDepth_", i, ".csv"), row.names = FALSE)
}

# Generate Predictions
predGBM = h2o.predict(object = TopModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_26102016_01.csv", row.names = FALSE)
# THE SUBMISSION SCORED ~ 1173.XXXXX - NOT THE BEST. MORE TUNING REQUIRED.


-------------------------------------------------------------------------------------
##### GBM: # OF TREES TUNING                                                    #####
-------------------------------------------------------------------------------------

HyperParam = list(ntrees = c(100,200,400,800,1600,3200,6400,12800))

gridTreesSearch <- h2o.grid(algorithm = "gbm", 
                            x = IndAttrib,
                            y = DepAttrib, 
                            training_frame = ModTrain.hex, 
                            grid_id = "GRID_TOT_TREES",
                            hyper_params = HyperParam,
                            validation_frame = ModTest.hex, 
                            seed = 1, 
                            distribution = "gaussian",
                            max_depth = 10,
                            ## smaller learning rate is better
                            ## Due to learning_rate_annealing, we can start with a bigger learning rate
                            learn_rate = 0.05,                                                         
                            ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                            learn_rate_annealing = 0.95,
                            ## Early stopping configuration
                            stopping_rounds = 3, 
                            stopping_tolerance = 0.0001,
                            stopping_metric = "MSE",
                            ## sample 80% of rows per tree
                            sample_rate = 0.8,
                            ## sample 80% of columns per split
                            col_sample_rate = 0.8,
                            ## score every 10 trees to make early stopping reproducible
                            score_tree_interval = 5)

# Get grid:
gridTrees = h2o.getGrid(grid_id = "GRID_TOT_TREES", sort_by = "mse")

# Check error metrics for the models
lapply(gridTrees@model_ids, function(x) {h2o.mae(object = h2o.getModel(x), train = TRUE, valid = TRUE)})
lapply(gridTrees@model_ids, function(x) {h2o.rmse(object = h2o.getModel(x), train = TRUE, valid = TRUE)})

# Check the spread between Valid and Train metrics: MAE and RMSE
lapply(lapply(gridTrees@model_ids, function(x) {h2o.mae(object = h2o.getModel(x), train = TRUE, valid = TRUE)}), function(y){y["valid"] - y["train"]})

lapply(lapply(gridTrees@model_ids, function(x) {h2o.rmse(object = h2o.getModel(x), train = TRUE, valid = TRUE)}), function(y){y["valid"] - y["train"]})

### OBSERVATION: BASED ON gridTrees - BUILDING >200 TREES IS NOT RESULTING IN ANY IMPORVED ACCURACY/MAE/MSE REDUCTION. SUBSEQUENT TUNING ATTEMPT TO FIND # OF TREES IN MORE GRANULAR FASHION.

HyperParam = list(ntrees = seq(40,300,20))

gridTreesSearch2 <- h2o.grid(algorithm = "gbm", 
                            x = IndAttrib,
                            y = DepAttrib, 
                            training_frame = ModTrain.hex, 
                            grid_id = "GRID_TOT_TREES_2",
                            hyper_params = HyperParam,
                            validation_frame = ModTest.hex, 
                            seed = 1,
                            distribution = "gaussian",
                            max_depth = 10,
                            ## smaller learning rate is better
                            ## Due to learning_rate_annealing, we can start with a bigger learning rate
                            learn_rate = 0.05,                                                         
                            ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                            learn_rate_annealing = 0.95,
                            ## Early stopping configuration
                            stopping_rounds = 3, 
                            stopping_tolerance = 0.0001,
                            stopping_metric = "MSE",
                            ## sample 80% of rows per tree
                            sample_rate = 0.8,
                            ## sample 80% of columns per split
                            col_sample_rate = 0.8,
                            ## score every 10 trees to make early stopping reproducible
                            score_tree_interval = 5)

# Get grid:
gridTrees = h2o.getGrid(grid_id = "GRID_TOT_TREES_2", sort_by = "mse")

# Check error metrics for the models
lapply(gridTrees@model_ids, function(x) {h2o.mae(object = h2o.getModel(x), train = TRUE, valid = TRUE)})
lapply(gridTrees@model_ids, function(x) {h2o.rmse(object = h2o.getModel(x), train = TRUE, valid = TRUE)})

# Check the spread between Valid and Train metrics: MAE and RMSE
lapply(lapply(gridTrees@model_ids, function(x) {h2o.mae(object = h2o.getModel(x), train = TRUE, valid = TRUE)}), function(y){y["valid"] - y["train"]})

lapply(lapply(gridTrees@model_ids, function(x) {h2o.rmse(object = h2o.getModel(x), train = TRUE, valid = TRUE)}), function(y){y["valid"] - y["train"]})

### "NTREE" TUNING OUTCOME:
###     MAE/RMSE are minimized for 180 trees @ depth = 10 and learn_rate = 0.05 with 5% annealing
###     NTREE  RANGE (STRICT): 50-180
###     NTREE  RANGE (RELAXED): 50-250

TopModel = h2o.getModel(model_id = gridTrees@model_ids[[1]])

# Save the features from all 10 models

ImpVar = h2o.varimp(TopModel)
write.csv(x = as.data.frame(x = ImpVar), file = "FEATURES_GBM_GRID_NTree.csv", row.names = FALSE)

# Generate Predictions
predGBM = h2o.predict(object = TopModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_28102016_01.csv", row.names = FALSE)

### TOP MODEL SCORE: 1387.43941
### HIGHER NTREE MIGHT BE REQUIRED WITH LOWER LEARNING RATE. ALSO DEPTH SEEM TO HAVE A HIGHER IMPACT.


-------------------------------------------------------------------------------------
##### GBM: LEARNING RATE TUNING (NO ANNEALING)                                  #####
-------------------------------------------------------------------------------------
    
HyperParam = list(learn_rate = seq(0.01,0.2,0.01))

gridLRate <-       h2o.grid(algorithm = "gbm", 
                            x = IndAttrib,
                            y = DepAttrib, 
                            training_frame = ModTrain.hex, 
                            grid_id = "GRID_LEARNRATE_NOANNEAL",
                            hyper_params = HyperParam,
                            validation_frame = ModTest.hex, 
                            seed = 1, 
                            distribution = "gaussian",
                            max_depth = 10,
                            ntrees = 200,
                            ## smaller learning rate is better
                            ## Due to learning_rate_annealing, we can start with a bigger learning rate
                            #learn_rate = 0.05,                                                         
                            ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                            #learn_rate_annealing = 0.95,
                            ## Early stopping configuration
                            stopping_rounds = 3, 
                            stopping_tolerance = 0.0001,
                            stopping_metric = "MSE",
                            ## sample 80% of rows per tree
                            sample_rate = 0.8,
                            ## sample 80% of columns per split
                            col_sample_rate = 0.8,
                            ## score every 10 trees to make early stopping reproducible
                            score_tree_interval = 5)

# tuning iteration 2: RandomDiscrete Tuning

HyperParam = list(learn_rate = seq(0.01,0.15,0.01), 
                  ntrees = c(900,1200))

gridLRate3 <-       h2o.grid(algorithm = "gbm", 
                            x = IndAttrib,
                            y = DepAttrib, 
                            training_frame = ModTrain.hex, 
                            grid_id = "GRID_LEARNRATE_NOANNEAL_3",
                            hyper_params = HyperParam,
                            validation_frame = ModTest.hex, 
                            seed = 1, 
                            distribution = "gaussian",
                            max_depth = 10,
                            #ntrees = 200,
                            ## smaller learning rate is better
                            ## Due to learning_rate_annealing, we can start with a bigger learning rate
                            #learn_rate = 0.05,                                                         
                            ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                            #learn_rate_annealing = 0.95,
                            ## Early stopping configuration
                            ##stopping_rounds = 3, 
                            ##stopping_tolerance = 0.0001,
                            ##stopping_metric = "MSE",
                            ## sample 80% of rows per tree
                            sample_rate = 0.8,
                            ## sample 80% of columns per split
                            col_sample_rate = 0.8,
                            ## score every 5 trees to make early stopping reproducible
                            score_tree_interval = 5,
                            ## Search Strategy for Hyperparameter space
                            search_criteria =  list(strategy = "RandomDiscrete", 
                                                    max_runtime_secs = 12600,
                                                    stopping_rounds = 5, 
                                                    stopping_tolerance = 0.0001,
                                                    stopping_metric = "deviance"))


G1 <- h2o.getGrid("GRID_LEARNRATE_NOANNEAL_3",sort_by = "mse")

M1 <- h2o.getModel(model_id = G1@model_ids[[1]])

lapply(lapply(G1@model_ids, function(x){h2o.getModel(model_id = x)}), 
       function(M1){M1@model$scoring_history$number_of_trees[which.min(M1@model$scoring_history$validation_mae)]})

#-------------------------------------------------------
### Stratified Tunning For NTrees: Cartesian Grid Search
#-------------------------------------------------------

### Strata 1: 0.00-0.05
HyperParam = list(learn_rate = seq(0.005,0.05,0.005), 
                  ntrees = c(600))

gridLRate4 <-       h2o.grid(algorithm = "gbm", 
                             x = IndAttrib,
                             y = DepAttrib, 
                             training_frame = ModTrain.hex, 
                             grid_id = "GRID_LEARNRATE_NOANNEAL_4",
                             hyper_params = HyperParam,
                             validation_frame = ModTest.hex, 
                             seed = 1, 
                             distribution = "gaussian",
                             max_depth = 10,
                             #ntrees = 200,
                             ## smaller learning rate is better
                             ## Due to learning_rate_annealing, we can start with a bigger learning rate
                             #learn_rate = 0.05,                                                         
                             ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                             #learn_rate_annealing = 0.95,
                             ## Early stopping configuration
                             stopping_rounds = 5, 
                             stopping_tolerance = 0.0001,
                             stopping_metric = "deviance",
                             ## sample 80% of rows per tree
                             sample_rate = 0.8,
                             ## sample 80% of columns per split
                             col_sample_rate = 0.8,
                             ## score every 5 trees to make early stopping reproducible
                             score_tree_interval = 5)

### Strata 1: 0.055-0.100
HyperParam = list(learn_rate = seq(0.055,0.100,0.005), 
                  ntrees = c(300))

gridLRate5 <-       h2o.grid(algorithm = "gbm", 
                             x = IndAttrib,
                             y = DepAttrib, 
                             training_frame = ModTrain.hex, 
                             grid_id = "GRID_LEARNRATE_NOANNEAL_5",
                             hyper_params = HyperParam,
                             validation_frame = ModTest.hex, 
                             seed = 1, 
                             distribution = "gaussian",
                             max_depth = 10,
                             #ntrees = 200,
                             ## smaller learning rate is better
                             ## Due to learning_rate_annealing, we can start with a bigger learning rate
                             #learn_rate = 0.05,                                                         
                             ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                             #learn_rate_annealing = 0.95,
                             ## Early stopping configuration
                             stopping_rounds = 5, 
                             stopping_tolerance = 0.0001,
                             stopping_metric = "deviance",
                             ## sample 80% of rows per tree
                             sample_rate = 0.8,
                             ## sample 80% of columns per split
                             col_sample_rate = 0.8,
                             ## score every 5 trees to make early stopping reproducible
                             score_tree_interval = 5)

### BUILDING THE BEST MODEL & GIVE IT A TRY FOR KAGGLE SUBMISSION
### BEST MODEL PARAMETER DERIVED BASED ON THE HYPER PARAMETERS TUNED IN LAST 4 GRIDS

HyperParam = list(learn_rate = 0.02, ntrees = 600, max_depth = 10)

KaggelModel1 <-       h2o.grid(algorithm = "gbm", 
                             x = IndAttrib,
                             y = DepAttrib, 
                             training_frame = ModTrain.hex, 
                             grid_id = "KAGGLE_MODEL_1",
                             hyper_params = HyperParam,
                             validation_frame = ModTest.hex, 
                             seed = 1, 
                             distribution = "gaussian",
                             stopping_rounds = 5, 
                             stopping_tolerance = 0.0001,
                             stopping_metric = "deviance",
                             ## sample 80% of rows per tree
                             sample_rate = 0.8,
                             ## sample 80% of columns per split
                             col_sample_rate = 0.8,
                             ## score every 5 trees to make early stopping reproducible
                             score_tree_interval = 5)


TopModel = h2o.getModel(model_id = KaggelModel1@model_ids[[1]])
# Generate Predictions
predGBM = h2o.predict(object = TopModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_29102016_01.csv", row.names = FALSE)

### THIS SUBMISSION SCORED: 1158.XXXX ON KAGGLE | MAE BENCHMARK: 0.423925


### FURTHER TUNING OF LEARNING RATE AT DEPTH = 10 | AFTER KAGGLE SUBMISSION

### Strata 1: 0.055-0.100
HyperParam = list(learn_rate = seq(0.005,0.040,0.0025), 
                  ntrees = c(1200))

gridLRate5 <-       h2o.grid(algorithm = "gbm", 
                             x = IndAttrib,
                             y = DepAttrib, 
                             training_frame = ModTrain.hex, 
                             grid_id = "GRID_LEARNRATE_NOANNEAL_6",
                             hyper_params = HyperParam,
                             validation_frame = ModTest.hex, 
                             seed = 1, 
                             distribution = "gaussian",
                             max_depth = 10,
                             #ntrees = 200,
                             ## smaller learning rate is better
                             ## Due to learning_rate_annealing, we can start with a bigger learning rate
                             #learn_rate = 0.05,                                                         
                             ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                             #learn_rate_annealing = 0.95,
                             ## Early stopping configuration
                             stopping_rounds = 5, 
                             stopping_tolerance = 0.0001,
                             stopping_metric = "deviance",
                             ## sample 80% of rows per tree
                             sample_rate = 0.8,
                             ## sample 80% of columns per split
                             col_sample_rate = 0.8,
                             ## score every 5 trees to make early stopping reproducible
                             score_tree_interval = 5)

### For Kaggle Submission: New Model based on the last round of tuning

HyperParam = list(learn_rate = 0.0125, ntrees = 1000, max_depth = 10)

KaggelModel1 <-       h2o.grid(algorithm = "gbm", 
                               x = IndAttrib,
                               y = DepAttrib, 
                               training_frame = ModTrain.hex, 
                               grid_id = "KAGGLE_MODEL_1",
                               hyper_params = HyperParam,
                               validation_frame = ModTest.hex, 
                               seed = 1, 
                               distribution = "gaussian",
                               stopping_rounds = 5, 
                               stopping_tolerance = 0.0001,
                               stopping_metric = "deviance",
                               ## sample 80% of rows per tree
                               sample_rate = 0.8,
                               ## sample 80% of columns per split
                               col_sample_rate = 0.8,
                               ## score every 5 trees to make early stopping reproducible
                               score_tree_interval = 5)


TopModel = h2o.getModel(model_id = KaggelModel1@model_ids[[1]])
# Generate Predictions
predGBM = h2o.predict(object = TopModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_29102016_02.csv", row.names = FALSE)

## BEST KAGGLE SUBMISSION SO FAR - SCORED: 1156.34872 | BENCHMARK MAE: 0.423762

### BUILDING A MODEL WITH 5 & 10 DEPTH TO SEE THE IMPACT

HyperParam = list(learn_rate = 0.0125, ntrees = 1000, max_depth = c(5,15))

KaggelModel2 <-       h2o.grid(algorithm = "gbm", 
                               x = IndAttrib,
                               y = DepAttrib, 
                               training_frame = ModTrain.hex, 
                               grid_id = "KAGGLE_MODEL_2",
                               hyper_params = HyperParam,
                               validation_frame = ModTest.hex, 
                               seed = 1, 
                               distribution = "gaussian",
                               stopping_rounds = 5, 
                               stopping_tolerance = 0.0001,
                               stopping_metric = "deviance",
                               ## sample 80% of rows per tree
                               sample_rate = 0.8,
                               ## sample 80% of columns per split
                               col_sample_rate = 0.8,
                               ## score every 5 trees to make early stopping reproducible
                               score_tree_interval = 5)

### BUILDING A MODEL WITH VARIOUS DEPTH & CONSTANT LEARNING RATE

#HyperParam = list(learn_rate = 0.0125, ntrees = 1500, max_depth = c(9,11,12,13,14,15,16,17,18))
HyperParam = list(learn_rate = 0.02, ntrees = 1200, max_depth = c(4,5,6,7,8,9,10,11,12,13))

KaggelGrid4 <-       h2o.grid(algorithm = "gbm", 
                               x = IndAttrib,
                               y = DepAttrib, 
                               training_frame = ModTrain.hex, 
                               grid_id = "KAGGLE_GRID_4",
                               hyper_params = HyperParam,
                               validation_frame = ModTest.hex, 
                               seed = 1, 
                               distribution = "gaussian",
                               stopping_rounds = 5, 
                               stopping_tolerance = 0.0001,
                               stopping_metric = "deviance",
                               ## sample 80% of rows per tree
                               sample_rate = 0.8,
                               ## sample 80% of columns per split
                               col_sample_rate = 0.8,
                               ## score every 5 trees to make early stopping reproducible
                               score_tree_interval = 5)
                               #search_criteria = list(strategy = "RandomDiscrete",
                                #                     max_runtime_secs = 15000))


KaggleGrid = h2o.getGrid("KAGGLE_GRID_4", sort_by = "mse")
#TopModel = h2o.getModel(model_id = KaggleGrid@model_ids[[1]])       # MAX_DEPTH = 7
TopModel = h2o.getModel(model_id = KaggleGrid@model_ids[[2]])       # MAX_DEPTH = 8
# Generate Predictions
predGBM = h2o.predict(object = TopModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_01112016_02.csv", row.names = FALSE)


### BEST KAGGLE SUBMISSION SO FAR - SCORED: 1154.11922 | BENCHMARK MAE: 0.4234688

## Trying learning rate annealing.
HyperParam = list(learn_rate = 0.02, ntrees = 2500, max_depth = c(7), learn_rate_annealing = c(0.999, 0.997, 0.995, 0.990, 0.980, 0.950, 0.900))

KaggleGrid5 <-       h2o.grid(algorithm = "gbm", 
                              x = IndAttrib,
                              y = DepAttrib, 
                              training_frame = ModTrain.hex, 
                              grid_id = "GBM_GRID_LR_ANNEAL_1",
                              hyper_params = HyperParam,
                              validation_frame = ModTest.hex, 
                              seed = 1, 
                              distribution = "gaussian",
                              stopping_rounds = 5, 
                              stopping_tolerance = 0.0001,
                              stopping_metric = "deviance",
                              ## sample 80% of rows per tree
                              sample_rate = 0.8,
                              ## sample 80% of columns per split
                              col_sample_rate = 0.8,
                              ## score every 5 trees to make early stopping reproducible
                              score_tree_interval = 5)


KaggleGrid = h2o.getGrid("GBM_GRID_LR_ANNEAL_1", sort_by = "mse")
TopModel = h2o.getModel(model_id = KaggleGrid@model_ids[[1]])       # MAX_DEPTH = 8
# Generate Predictions
predGBM = h2o.predict(object = TopModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_01112016_03.csv", row.names = FALSE)
### BEST KAGGLE SUBMISSION SO FAR - SCORED: 1154.00009 | BENCHMARK MAE: 0.423455

#####################################################################################
### LEARNING RATE ANNEALING:
###     1. High values of rate annealing has a big impact on model learning.
###             E.g. Annealing rate of 0.90 results in a very less learning esp. if the starting rate is                  too small. The learning rate decays very rapidly resulting in no learning.
###     2. For a smaller initial learning rate:
###             STRICTLY: Choose learning rate annealing in the range of 0.995-0.999
###             LOOSLY: Choose learning rate annealing in the range of 0.990-0.999
#####################################################################################

### TRYING HIGHER LEARNING RATES FOR FASTER CONVEREGENCE

HyperParam = list(learn_rate = 0.03, ntrees = 1200, max_depth = c(4,5,6,7,8,9,10), learn_rate_annealing = c(1,0.999,0.998))

KaggleGrid6 <-       h2o.grid(algorithm = "gbm", 
                              x = IndAttrib,
                              y = DepAttrib, 
                              training_frame = ModTrain.hex, 
                              grid_id = "GRID_LR_0_PT_03",
                              hyper_params = HyperParam,
                              validation_frame = ModTest.hex, 
                              seed = 1, 
                              distribution = "gaussian",
                              stopping_rounds = 5, 
                              stopping_tolerance = 0.0001,
                              stopping_metric = "deviance",
                              ## sample 80% of rows per tree
                              sample_rate = 0.8,
                              ## sample 80% of columns per split
                              col_sample_rate = 0.8,
                              score_tree_interval = 5)

KaggleGrid = h2o.getGrid("GRID_LR_0_PT_03", sort_by = "mse")
TopModel = h2o.getModel(model_id = KaggleGrid@model_ids[[1]])       # MAX_DEPTH = 8
# Generate Predictions
predGBM = h2o.predict(object = TopModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predGBM = h2o.exp(predGBM)
dfGBMPredictions = as.data.frame(h2o.cbind(TestId, predGBM))
names(dfGBMPredictions) = c("id", "loss")
write.csv(x = dfGBMPredictions, file = "H2O_GBM_02112016_01.csv", row.names = FALSE)

##########################################################################################
#### MAE BASED ON ORIGINAL FEATURES HAVE PLATUED-NEED TO EXPLORE BETTER FEATURES WITH GBM
#### AS OF 2-NOV-2016 REFOCUSSING ON INSOFE PROJECT
##########################################################################################
