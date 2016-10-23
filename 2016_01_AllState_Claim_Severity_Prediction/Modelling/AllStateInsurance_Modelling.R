##### EXPERIMENT WITH KAGGLE DATA SET MODELLING

##### FEATURE ENGINEERING: NONE
##### FRAMEWORK: H2O

rm(list = ls(all.names = TRUE))
library(h2o)

# Initialize H2O
h2o.init(nthreads = 3, max_mem_size = "1G")

# Read Train & Test into H2O
AllStateTrain.hex = h2o.uploadFile(path = "C:/02 KAGGLE/2016_AllState_Claim_Severity_Prediction/train.csv", destination_frame = "AllStateTrain.hex", header = TRUE)
AllStateTest.hex = h2o.uploadFile(path = "C:/02 KAGGLE/2016_AllState_Claim_Severity_Prediction/test.csv", destination_frame = "AllStateTest.hex", header = TRUE)

AllStateTrain.hex$id = NULL

SplitFrames = h2o.splitFrame(data = AllStateTrain.hex, ratios = 0.7, seed = 2016)
ModTrain.hex = h2o.assign(data = SplitFrames[[1]], key = "ModTrain.hex")
ModTest.hex = h2o.assign(data = SplitFrames[[2]], key = "ModTest.hex")

DepAttrib = "loss"
IndAttrib = setdiff(names(ModTrain.hex), DepAttrib)


################################################################################
##### GLM with Lasso (L1), Ridge (L2) & ElasticNet @ Alpha = 0.5           #####
################################################################################

glmRidge = h2o.glm(x = IndAttrib, y = DepAttrib, training_frame = ModTrain.hex, model_id = "AllState_Ridge_01", family = "gaussian", alpha = 0, lambda_search = TRUE, validation_frame = ModTest.hex)

glmENet = h2o.glm(x = IndAttrib, y = DepAttrib, training_frame = ModTrain.hex, model_id = "AllState_ElasticNet_01", family = "gaussian", alpha = 0.5, lambda_search = TRUE, validation_frame = ModTest.hex)

glmLasso = h2o.glm(x = IndAttrib, y = DepAttrib, training_frame = ModTrain.hex, model_id = "AllState_Lasso_01", family = "gaussian", alpha = 1, lambda_search = TRUE, validation_frame = ModTest.hex)

### MAE calculation

h2o.mae(object = glmRidge, train = TRUE, valid = TRUE)
h2o.mae(object = glmENet, train = TRUE, valid = TRUE)
h2o.mae(object = glmLasso, train = TRUE, valid = TRUE)

### Generate Predictions

predRidge = h2o.predict(object = glmRidge, newdata = AllStateTest.hex)
predENet = h2o.predict(object = glmENet, newdata = AllStateTest.hex)
predLasso = h2o.predict(object = glmLasso, newdata = AllStateTest.hex)

### Convert all predictions to R DataFrames

dfRidge = as.data.frame(h2o.cbind(AllStateTest.hex$id, predRidge))
names(dfRidge) = c("id", "loss")

dfEnet = as.data.frame(h2o.cbind(AllStateTest.hex$id, predENet))
names(dfEnet) = c("id", "loss")

dfLasso = as.data.frame(h2o.cbind(AllStateTest.hex$id, predLasso))
names(dfLasso) = c("id", "loss")

### Write all predictions into a CSV

write.csv(x = dfRidge, file = "H2O_Ridge.csv", row.names = FALSE)
write.csv(x = dfEnet, file = "H2O_Enet.csv", row.names = FALSE)
write.csv(x = dfLasso, file = "H2O_Lasso.csv", row.names = FALSE)


################################################################################
##### Grid Search for the best GLM - with all variables                    #####
################################################################################

### Allowing Lasso & Ridge Regualarization To Be Included In GRID SEARCH

lamdas = c(10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001)
Alpha = seq(0,1,0.1)
HParam = list(alpha = Alpha, lambda = lamdas)

glmGridSearch = h2o.grid(algorithm = "glm", 
                         grid_id = "GLMGrid_01", 
                         hyper_params = HParam, 
                         x = IndAttrib,
                         y = DepAttrib,
                         training_frame = ModTrain.hex,
                         family = "gaussian",
                         is_supervised = TRUE)

glmGridModels = lapply(glmGridSearch@model_ids, function(x){h2o.getModel(x)})
MAE = unlist(lapply(glmGridModels, function(x) { h2o.mae(object = x)}))
BestGridModel = glmGridModels[[which.min(MAE)]]

### Generate Predictions From The Best Grid Model & Write To CSV
predGridGLM = h2o.predict(object = BestGridModel, newdata = AllStateTest.hex)
dfGridGLM = as.data.frame(h2o.cbind(AllStateTest.hex$id, predGridGLM))
names(dfGridGLM) = c("id", "loss")
write.csv(x = dfGridGLM, file = "H2O_GridGLM.csv", row.names = FALSE)


### NOT allowing Lasso & Ridge Regualarization To Be Included In GRID SEARCH
### Narrowed down the values for "lambdas" and expanded the values for Alpha

Lamdas = c(0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001)
Alpha = seq(0.5,0.95,0.05)
HParam = list(alpha = Alpha, lambda = Lamdas)

glmGridSearch2 = h2o.grid(algorithm = "glm", 
                         grid_id = "GLMGrid_02", 
                         hyper_params = HParam, 
                         x = IndAttrib,
                         y = DepAttrib,
                         training_frame = ModTrain.hex,
                         validation_frame = ModTest.hex,
                         family = "gaussian",
                         is_supervised = TRUE)

glmGridModels2 = lapply(glmGridSearch2@model_ids, function(x){h2o.getModel(x)})
MAE2 = unlist(lapply(glmGridModels2, function(x) {h2o.mae(object = x)}))
BestGridModel2 = glmGridModels2[[which.min(MAE2)]]

### Generate Predictions From The Best Grid Model & Write To CSV
predGridGLM2 = h2o.predict(object = BestGridModel2, newdata = AllStateTest.hex)
dfGridGLM2 = as.data.frame(h2o.cbind(AllStateTest.hex$id, predGridGLM2))
names(dfGridGLM2) = c("id", "loss")
write.csv(x = dfGridGLM2, file = "H2O_GridGLM_2.csv", row.names = FALSE)


################################################################################
#### Trying Cross Validation with Lasso Regression                         #####
################################################################################

glmLasso2 = h2o.glm(x = IndAttrib, 
                    y = DepAttrib, 
                    training_frame = AllStateTrain.hex, 
                    model_id = "AllState_Lasso_02", 
                    family = "gaussian", alpha = 1, 
                    nfolds = 10)

h2o.mae(object = glmLasso2, train = TRUE)


#### GBM Model: BASIC GBM MODEL - BEST ACCURACY SO FAR (MODEL DIRECTLY CREATED IN H2O FLOW)

GBM1 <- h2o.getModel(model_id = "gbm-627f0100-b75d-497a-943b-59227d449492")
h2o.mae(object = GBM1)
### Generate Predictions From The Best Grid Model & Write To CSV
predGBM1 = h2o.predict(object = GBM1, newdata = AllStateTest.hex)
dfGBM1 = as.data.frame(h2o.cbind(AllStateTest.hex$id, predGBM1))
names(dfGBM1) = c("id", "loss")
write.csv(x = dfGBM1, file = "H2O_GBM1.csv", row.names = FALSE)


################################################################################
##### DISTRIBUTED RANDOM FOREST WITH H20                                   #####
################################################################################

##### GRID SEARCH MODEL_01: 80-100 Min of Grid Search

HypParams = list(ntrees = c(50,75,100),
                 max_depth = c(18,20,22))

drfGridSearch1 = h2o.grid(algorithm = "randomForest", 
                          grid_id = "DRF_GRID_01", 
                          hyper_params = HypParams, 
                          x = IndAttrib,
                          y = DepAttrib,
                          training_frame = ModTrain.hex,
                          validation_frame = ModTest.hex,
                          is_supervised = TRUE,
                          seed = 2016)

# BEST MODEL: ntree = 100 | max_depth = 22 | seed = 100 | features = ALL
drfGrid1 <- h2o.getGrid(grid_id = "DRF_GRID_01", sort_by = "mse", decreasing = FALSE)
print(drfGrid1)
drfBestGridModel1 <- h2o.getModel(drfGrid1@model_ids[[1]])
h2o.mae(object = drfBestGridModel1, train = TRUE, valid = TRUE)

# Generate Predictions
predDRF1 = h2o.predict(object = drfBestGridModel1, newdata = AllStateTest.hex)
dfDRF1 = as.data.frame(h2o.cbind(AllStateTest.hex$id, predDRF1))
names(dfDRF1) = c("id", "loss")
write.csv(x = dfDRF1, file = "H2O_DRF_01.csv", row.names = FALSE)

### Selecting important attribites from the best DRF model [ScaledImportance > 5%]
impVar = h2o.varimp(drfBestGridModel1)
impAttribs = impVar[impVar$scaled_importance >= 0.05,]$variable

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

GBM2 <- h2o.gbm(x = impAttribs, 
                y = DepAttrib, 
                training_frame = ModTrain.hex, 
                model_id = "GBM_MODEL_02", 
                validation_frame = ModTest.hex, 
                ntrees = 350, 
                seed = 2016, 
                distribution = "gaussian", 
                max_depth = 5, 
                stopping_rounds = 3, 
                stopping_metric = "MSE")

h2o.mae(object = GBM2, train = TRUE, valid = TRUE)
h2o.varimp_plot(model = GBM2)

# Generate Predictions
predDRF1 = h2o.predict(object = GBM2, newdata = AllStateTest.hex)
dfDRF1 = as.data.frame(h2o.cbind(AllStateTest.hex$id, predDRF1))
names(dfDRF1) = c("id", "loss")
write.csv(x = dfDRF1, file = "H2O_GBM_04.csv", row.names = FALSE)


