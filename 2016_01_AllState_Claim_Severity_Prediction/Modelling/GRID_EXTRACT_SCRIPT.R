# LIST OF GRIDS FROM WHICH DATA NEEDS TO BE EXTRACTED
GRID_Ids = c("GRID_LEARNRATE_NOANNEAL_2", "GRID_LEARNRATE_NOANNEAL_3", "GRID_LEARNRATE_NOANNEAL_4", "GRID_LEARNRATE_NOANNEAL_5", "GRID_LEARNRATE_NOANNEAL_6")

# EMPTY DATA FRAMES FOR CONSOLIDATING DATA
dfFinal = data.frame()
dfImpVars = data.frame()

for(i in 1:length(GRID_Ids))
{

  # OBTAIN GRID AND "THE BEST" MODEL OF THAT GRID
  GRID = h2o.getGrid(grid_id = GRID_Ids[i], sort_by = "mse")
  TopMod = h2o.getModel(model_id = GRID@model_ids[[1]])
  
  ModelName = unlist(GRID@model_ids)
  # EXTRACT MODEL PARAMETERS FOR EACH MODEL IN THE GRID
  ntrees = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$ntrees}))
  max_depth = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$max_depth}))
  learn_rate = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$learn_rate}))
  learn_rate_annealing = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$learn_rate_annealing}))
  col_sample_rate = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$col_sample_rate}))
  stopping_rounds = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$stopping_rounds}))
  stopping_metric = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$stopping_metric}))
  stopping_tolerance = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$stopping_tolerance}))
  score_tree_interval  = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$score_tree_interval}))
  distribution = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), function(y){y@allparameters$distribution}))
  
  # EXTRACT IMPORTANT PARAMETERS FOR SCORING ROUND OF EACH MODEL IN THE GRID
  Min_Val_Dev = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), 
                              function(y){min(y@model$scoring_history[,"validation_deviance"])}))
  TrainDev_At_Min_Val_Dev = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), 
                                          function(y){y@model$scoring_history$training_deviance[which.min(y@model$scoring_history[,"validation_deviance"])]}))
  ntree_At_Min_Val_Dev = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), 
                                    function(y){y@model$scoring_history$number_of_trees[which.min(y@model$scoring_history[,"validation_deviance"])]}))
  Min_Val_MAE = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), 
                              function(y){min(y@model$scoring_history[,"validation_mae"])}))
  TrainMAE_At_Min_Val_MAE = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), 
                                          function(y){y@model$scoring_history$training_mae[which.min(y@model$scoring_history[,"validation_mae"])]}))
  ntree_At_Min_Val_MAE = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), 
                                       function(y){y@model$scoring_history$number_of_trees[which.min(y@model$scoring_history[,"validation_mae"])]}))
  MaxTreesBuilt = unlist(lapply(lapply(GRID@model_ids, function(x){h2o.getModel(x)}), 
                                function(y){max(y@model$scoring_history$number_of_trees)}))
  
  # COMBINE EXTRACTED RESULTS VERTICALLY
  df1 = cbind(ModelName, ntrees, max_depth, learn_rate, learn_rate_annealing, col_sample_rate, stopping_rounds, stopping_metric, stopping_tolerance, 
              score_tree_interval, distribution, Min_Val_Dev, TrainDev_At_Min_Val_Dev, ntree_At_Min_Val_Dev, Min_Val_MAE, TrainMAE_At_Min_Val_MAE,
              ntree_At_Min_Val_MAE, MaxTreesBuilt)
  
  # UPDATE FINAL DATA FRAMES: MODEL PARAMETERS + SCORING DATA + IMPORTANT VARIABLS
  dfFinal = rbind(dfFinal, as.data.frame(df1))
  dfImpVars = rbind(dfImpVars, TopMod@model$variable_importances)
}

# WRITE TO DISK
write.csv(x = dfFinal, file = "GBM_TUNING_PARAMETERS.csv", row.names = FALSE)
write.csv(x = dfImpVars, file = "GBM_TOP_MODEL_VARIABLE_IMPORTANCE.csv", row.names = FALSE)