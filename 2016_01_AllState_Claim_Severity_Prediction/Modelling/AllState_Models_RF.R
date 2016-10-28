##### EXPERIMENT WITH KAGGLE DATA SET MODELLING

##### FEATURE ENGINEERING: NONE
##### FRAMEWORK: H2O

rm(list = ls(all.names = TRUE))
library(h2o)

# Initialize H2O
h2o.init(nthreads = -1, min_mem_size = "6G")

# Cleanup H2O Enviroment
#h2o.removeAll()

# Read Train & Test into H2O
AllStateTrain.hex = h2o.uploadFile(path = "C:/02 KAGGLE/2016_AllState_Claim_Severity_Prediction/train.csv", destination_frame = "AllStateTrain.hex", header = TRUE)
AllStateTest.hex = h2o.uploadFile(path = "C:/02 KAGGLE/2016_AllState_Claim_Severity_Prediction/test.csv", destination_frame = "AllStateTest.hex", header = TRUE)

AllStateTrain.hex$loss = h2o.log(AllStateTrain.hex$loss)

TrainId = AllStateTrain.hex$id
TestId = AllStateTest.hex$id
AllStateTrain.hex$id = NULL
AllStateTest.hex$id = NULL

# DATA TRANSFORMATION: Converting Loss varibale into Log10(loss) to make the Right Skewed distribution Normal
#ncol(AllStateTrain.hex)
#AllStateTrain.hex = AllStateTrain.hex[AllStateTrain.hex$loss >= 1, ]
#AllStateTrain.hex$LogLoss = h2o.log(AllStateTrain.hex$loss)
#ncol(AllStateTrain.hex)
# Selecting Only Subset of Training Data - Potentially Exlcuding Outliers

vFactors = paste0("cat", 1:116)
vNumbers = paste0("cont", 1:14)

PCA = h2o.prcomp(training_frame = AllStateTrain.hex, x = vNumbers, k = length(vNumbers), model_id = "PCA_01", transform = "STANDARDIZE", seed = 2016)

# Remove Continuous variable and add PCA Score - Train data frame
AllStateTrainPCA.hex = h2o.assign(data = h2o.predict(object = PCA, newdata = AllStateTrain.hex), key = "AllStateTrainPCA.hex")
AllStateTrain.hex[ ,vNumbers] = NULL
AllStateTrain.hex = h2o.cbind(AllStateTrain.hex, AllStateTrainPCA.hex[ ,1:12])

# Remove Continuous variable and add PCA Score - Test Data Frame
AllStateTestPCA.hex = h2o.assign(data = h2o.predict(object = PCA, newdata = AllStateTest.hex), key = "AllStateTestPCA.hex")
AllStateTest.hex[ ,vNumbers] = NULL
AllStateTest.hex = h2o.cbind(AllStateTest.hex, AllStateTestPCA.hex[ ,1:12])


SplitFrames = h2o.splitFrame(data = AllStateTrain.hex, ratios = 0.7, seed = 2016)
ModTrain.hex = h2o.assign(data = SplitFrames[[1]], key = "ModTrain.hex")
ModTest.hex = h2o.assign(data = SplitFrames[[2]], key = "ModTest.hex")

DepAttrib = "loss"
IndAttrib = setdiff(names(ModTrain.hex), DepAttrib)

################################################################################
##### DISTRIBUTED RANDOM FOREST - BASIC                                    #####
################################################################################

rfModel1 = h2o.randomForest(x = IndAttrib,y = DepAttrib, training_frame = ModTrain.hex, validation_frame = ModTest.hex, seed = 2016)

h2o.mae(object = rfModel1, train = TRUE, valid = TRUE)

--------------------------------------------------------------------------------------
################################################################################
##### DISTRIBUTED RANDOM FOREST WITH H20                                   #####
################################################################################

##### GRID SEARCH MODEL_01: 80-100 Min of Grid Search

#HypParams = list(ntrees = c(50,75,100), max_depth = c(18,20,22))
HypParams = list(ntrees = c(75,100,125), max_depth = c(20))

drfGridSearch1 = h2o.grid(algorithm = "randomForest", 
                          grid_id = "DRF_GRID_01", 
                          hyper_params = HypParams, 
                          x = IndAttrib,
                          y = DepAttrib,
                          training_frame = ModTrain.hex,
                          validation_frame = ModTest.hex,
                          mtries = 64,
                          is_supervised = TRUE,
                          seed = 2016)

# BEST MODEL: ntree = 100 | max_depth = 22 | seed = 100 | features = ALL
drfGrid1 <- h2o.getGrid(grid_id = "DRF_GRID_01", sort_by = "mse", decreasing = FALSE)
print(drfGrid1)
drfBestGridModel1 <- h2o.getModel(drfGrid1@model_ids[[1]])
h2o.mae(object = drfBestGridModel1, train = TRUE, valid = TRUE)

# Generate Predictions
predDRF1 = h2o.predict(object = drfBestGridModel1, newdata = AllStateTest.hex)
dfDRF1 = as.data.frame(h2o.cbind(TestId, predDRF1))
names(dfDRF1) = c("id", "loss")
write.csv(x = dfDRF1, file = "H2O_DRF_01.csv", row.names = FALSE)

### Selecting important attribites from the best DRF model [ScaledImportance > 5%]
impVar = h2o.varimp(drfBestGridModel1)
impAttribs = impVar[impVar$scaled_importance >= 0.01,]$variable

# Rebuilding the DRF with only important attributes - No Cross Validation
drfModel1 <- h2o.randomForest(x = impAttribs,
                              y = DepAttrib,
                              training_frame = ModTrain.hex, 
                              model_id = "DRF_MODEL_01", 
                              validation_frame = ModTest.hex, 
                              ntrees = 100, 
                              max_depth = 22, 
                              seed = 2016)

h2o.mae(object = drfModel1, train = TRUE, valid = TRUE)
h2o.varimp_plot(model = drfModel1)

# Generate Predictions
predDRF1 = h2o.predict(object = drfModel1, newdata = AllStateTest.hex)
dfDRF1 = as.data.frame(h2o.cbind(AllStateTest.hex$id, predDRF1))
names(dfDRF1) = c("id", "loss")
write.csv(x = dfDRF1, file = "H2O_DRF_02.csv", row.names = FALSE)

##### DRF with only important attributes: 10-fold Cross Validation
# COULD NOT USE
drfModel2 <- h2o.randomForest(x = impAttribs, y = DepAttrib, training_frame = AllStateTrain.hex, model_id = "DRF_MODEL_02", ntrees = 50, max_depth = 20, seed = 2016, nfolds = 10, stopping_rounds = 2, stopping_metric = "MSE", stopping_tolerance = 1e-3, score_tree_interval = 50)

h2o.mae(object = drfModel2, train = TRUE)


##### GBM

LR = c(0.2)
Trees = c(400,600,800,1000)
Depth = c(5)

HParam = list(learn_rate = LR, ntrees = Trees, max_depth = Depth)

#GBM2 <- h2o.gbm(x = IndAttrib, learn_rate = 0.2,
 #               y = DepAttrib, 
  #              training_frame = ModTrain.hex, 
   #             model_id = "GBM_MODEL_02", 
    #            validation_frame = ModTest.hex, 
     #           ntrees = 250, 
      #          seed = 1, 
       #         distribution = "gaussian", 
        #        max_depth = 5, 
         #       stopping_rounds = 3, 
          #      stopping_metric = "MSE")

GBM3 <- h2o.grid(algorithm = "gbm", 
                x = IndAttrib,
                y = DepAttrib, 
                training_frame = ModTrain.hex, 
                grid_id = "GBM_GRID_01",
                hyper_params = HParam,
                validation_frame = ModTest.hex, 
                seed = 1, 
                distribution = "gaussian", 
                stopping_rounds = 3, 
                stopping_metric = "MSE")

gridGBM = h2o.getGrid(grid_id = "GBM_GRID_01", sort_by = "mse")
Id = gridGBM@model_ids[[1]]
BestModel = h2o.getModel(model_id = Id)

BestModel@allparameters

h2o.mae(object = BestModel, train = TRUE, valid = TRUE)
h2o.varimp_plot(model = BestModel, num_of_features = 35)

# Generate Predictions
predDRF1 = h2o.predict(object = BestModel, newdata = AllStateTest.hex)
# Following step is required only for Log(loss) predictions
predDRF1 = h2o.exp(predDRF1)
dfDRF1 = as.data.frame(h2o.cbind(TestId, predDRF1))
names(dfDRF1) = c("id", "loss")
write.csv(x = dfDRF1, file = "H2O_GBM_03.csv", row.names = FALSE)





CALC_MAE <- function(Actuals, Predicted)
{
    ifelse((length(Actuals) != 0 & length(Predicted) != 0 & (length(Actuals) == length(Predicted))), return(sum(abs(Actuals - Predicted))/nrow(Actuals)), return(NaN))
}