#####################################################################################
##### PURPOSE:    REUSABLE MODELLING LIBRARY - USING H2O PLATFORM               #####
##### ASSUMPTION: CODE ASSUMES THAT H2O PLATFORM IS RUNNING                     #####
##### SCOPE:      LIMITED TO MODEL BULDING & SELECTION ONLY                     #####
#####################################################################################

##### H2O: GLM-LASSO-01                                                         #####
#####################################################################################

GLM_LASSO <- function(TrainFrame, TestFrame, ModFamily, IndAttrib, DepAttrib)
{
    glm_Lasso = h2o.glm(x = IndAttrib, 
                        y = DepAttrib, 
                        training_frame = TrainFrame,
                        validation_frame = TestFrame,
                        family = ModFamily, 
                        alpha = 1, 
                        lambda_search = TRUE,
                        remove_collinear_columns = TRUE)
    return(glm_Lasso)
}


##### H2O: GLM-LASSO-02                                                         #####
#####################################################################################

GLM_LASSO_CV <- function(CVFrame, ModFamily, IndAttrib, DepAttrib, NFolds)
{
    glm_Lasso = h2o.glm(x = IndAttrib, 
                        y = DepAttrib, 
                        training_frame = CVFrame,
                        family = ModFamily, 
                        alpha = 1, 
                        lambda_search = TRUE, 
                        nfolds = NFolds, 
                        keep_cross_validation_fold_assignment = TRUE,
                        remove_collinear_columns = TRUE)
    return(glm_Lasso)
}


##### H2O: GLM-RIDGE-01                                                         #####
#####################################################################################

GLM_RIDGE <- function(TrainFrame, TestFrame, ModFamily, IndAttrib, DepAttrib)
{
    glm_Ridge = h2o.glm(x = IndAttrib, 
                        y = DepAttrib, 
                        training_frame = TrainFrame,
                        validation_frame = TestFrame,
                        family = ModFamily, 
                        alpha = 0, 
                        lambda_search = TRUE,
                        remove_collinear_columns = TRUE)
    return(glm_Ridge)
}


##### H2O: GLM-RIDGE-02                                                         #####
#####################################################################################

GLM_RIDGE_CV <- function(CVFrame, ModFamily, IndAttrib, DepAttrib, NFolds)
{
    glm_Ridge = h2o.glm(x = IndAttrib,
                        y = DepAttrib, 
                        training_frame = CVFrame,
                        family = ModFamily, 
                        alpha = 0, 
                        lambda_search = TRUE, 
                        nfolds = NFolds,
                        keep_cross_validation_fold_assignment = TRUE,
                        remove_collinear_columns = TRUE)
    return(glm_Ridge)
}


##### H2O: GLM-ELASTICNET-01                                                    #####
#####################################################################################

GLM_EALSTIC_NET <- function(TrainFrame, TestFrame, ModFamily, IndAttrib, DepAttrib, Alpha=0.5)
{
    glm_Enet = h2o.glm(x = IndAttrib, 
                       y = DepAttrib, 
                       training_frame = TrainFrame,
                       validation_frame = TestFrame,
                       family = ModFamily,
                       alpha = Alpha, 
                       lambda_search = TRUE,
                       remove_collinear_columns = TRUE)
    return(glm_Ridge)
}


##### H2O: GLM-ELASTICNET-02                                                    #####
#####################################################################################

GLM_ELASTIC_NET_CV <- function(CVFrame, ModFamily, IndAttrib, DepAttrib, Alpha=0.5,NFolds)
{
    glm_Enet = h2o.glm(x = IndAttrib,
                       y = DepAttrib, 
                       training_frame = CVFrame,
                       family = ModFamily, 
                       alpha = Alpha, 
                       lambda_search = TRUE, 
                       nfolds = NFolds,
                       keep_cross_validation_fold_assignment = TRUE,
                       remove_collinear_columns = TRUE)
    return(glm_Enet)
}


##### H2O: GLM WITH GRID SEARCH                                                 #####
#####################################################################################

GLM_GRID_BASIC <- function(TrainFrame, TestFrame, ModFamily, IndAttrib, DepAttrib, Alpha=0.5)
{
    glm_Enet = h2o.glm(x = IndAttrib, 
                       y = DepAttrib, 
                       training_frame = TrainFrame,
                       validation_frame = TestFrame,
                       family = ModFamily,
                       alpha = Alpha, 
                       lambda_search = TRUE,
                       remove_collinear_columns = TRUE)
    return(glm_Ridge)
}