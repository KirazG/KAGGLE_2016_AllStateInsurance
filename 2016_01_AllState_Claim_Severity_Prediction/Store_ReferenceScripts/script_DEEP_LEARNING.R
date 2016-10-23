# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#---------------------------------------------------------------------------
# Learn more about H2O DeepLearning here
# https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/deeplearning
# --------------------------------------------------------------------------
# Load libraries
library(h2o); h2o.init(nthreads = -1)

# Read input data
train <- h2o.importFile("../input/train.csv", destination_frame = "train.hex")
test <- h2o.importFile("../input/test.csv", destination_frame = "test.hex")

# print dimension of dataset
dim(train); dim(test)

# split the training dataset into 80/20
splits <- h2o.splitFrame(train, c(0.8,0.199), seed=12345)

# select frames for H2o
train  <- h2o.assign(splits[[1]], "train80.hex") # 80%
valid  <- h2o.assign(splits[[2]], "valid20.hex") # 20%

# print dimension of dataset
dim(train); dim(valid)

# assign predictors(x) and response (y)
response <- "loss"
predictors <- setdiff(names(train), response)
predictors

# for this example we built a DNN (Input x Tanh x Tanh x Linear) 
# of dimension 1269 x 80 x 80 x 1
print("Start deep learning...")
beat_RF <- h2o.deeplearning(
	   x=predictors,			# assign x
	   y=response, 				# assign y
	   training_frame=train,	# assign training frame (80%)
	   validation_frame=valid,	# assign test frame (20%)
	   nfolds=0,				# Cross validation

	   activation="Tanh",		# activation function 
	   hidden=c(80,80),			# select hidden topology (smaller is faster)
	   epochs=120,				# add more to increase accuracy

	   score_training_samples=10000, 	# faster training (can be set to 0)
	   stopping_rounds=20,			    # increase to run longer more accurate
	   stopping_metric="AUTO", 		    # "MSE","logloss","r2","misclassification","auto"
	   stopping_tolerance=0.0001,		# stop if not getting better
	   max_runtime_secs=0,			    # dont run too long
	   overwrite_with_best_model=TRUE,  # use best model along the way
	   
	   distribution="tweedie", 		# gaussian, poisson and others
	   tweedie_power=1.5,			# set tweedie power
	   input_dropout_ratio=0,		# Input layer dropout ratio (can improve generalization, try 0.1 or 0.2)
	   l1=0, 				# L1 regularization (can add stability and improve generalization
	   l2=0)				# L2 regularization (can add stability and improve generalization
	  
# plot training summary          
summary(beat_RF)

# predict
submission <- test[, 1]
submission$loss <- h2o.predict(beat_RF, test)

# export predictions  
h2o.downloadCSV(submission, filename = "submission_beat_RF.csv")

### END