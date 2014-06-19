# R's GBM model for Higgs Boson Machine Learning Challenge
#
# By yr@Kaggle
#
# Implements two fold CV
# - Inner CV for choosing the best cutoff for maximizing AMS
# - Outer CV for estimating the performance using the best cutoff chosen
# This implementation is based on the code I wrote for Kaggle's Loan Default
# Prediction Competition at
# https://github.com/ChenglongChen/Kaggle_Loan_Default_Prediction
# but with a few revisions.


## 
rm(list=ls(all=TRUE))
gc(reset=TRUE)
par(mfrow=c(1,1))


## put all the packages here
require(data.table)
require(caret)
require(gbm)
require(Hmisc)
require(caTools)


## set the path to Higgs which contains 
# - ./Data/training.csv
# - ./Data/test.csv
setwd('path-to-Higgs')



######################
## Helper functions ##
######################

#### This function computes AUC and optionally plots the ROC curve
func_compute_AUC <- function(labels, scores, plotROC=FALSE){
  
  if(plotROC==TRUE){
    # use alg='ROC' for large data; 
    # NOTE that with plotROC=TRUE, alg will be set to "ROC" internally,
    # so no worry here
    par(mfrow=c(1,1))
    auc <- colAUC(scores, labels, plotROC=TRUE, alg='ROC')
    auc <- as.numeric(auc)
  }else{
    # if we don't want to plot ROC curve, we use the following 
    # fast method for computing AUC
    x1 = scores[labels==1]; n1 = length(x1)
    x2 = scores[labels==0]; n2 = length(x2)
    r = rank(c(x1, x2))
    # the following is the original version on web
    # auc = (sum(r[1:n1]) - n1*(n1+1)/2)/(n1*n2)
    # however to avoid the Warning for larger data: n1 * n2 : NAs produced by integer overflow
    # we use the following code
    auc = (sum(r[1:n1]) - n1*(n1+1)/2)/n1/n2
  }
  cat('AUC: ', auc, '\n', sep='')
  return(auc)
}


#### This function computes approximate median significance (AMS)
# for varying cutoffs
#  - true labels: y_true_label
#  - predicted probability: y_pred_prob
#  - weight: weight
#  - cutoffs: varying cutoffs
func_compute_AMS_cutoffs <- function(y_true_label, y_pred_prob, weight, cutoffs){
  
  #### This function computes (AMS) for a single cutoff
  func_compute_AMS_single_cutoff <- function(y_true_label, y_pred_prob, weight, cutoff){
    RankOrder <- rank(y_pred_prob, ties.method='random')
    top <- as.integer(floor(cutoff * length(y_pred_prob)))
    thresh <- y_pred_prob[which(RankOrder==top)]
    y_pred_label <- ifelse(y_pred_prob>thresh, 's', 'b')
    r <- func_compute_s_b(y_true_label, y_pred_label, weight)
    s <- r[[1]]
    b <- r[[2]]
    ams <- func_compute_AMS(s, b)
    return(ams)
  }
  
  ## We now compute f1-score for varying cutoffs
  AMS <- sapply(cutoffs, function(cutoff)
    func_compute_AMS_single_cutoff(y_true_label, y_pred_prob, weight, cutoff))
  return(AMS)
}


#### This function computes unnormalized true positive and false positive
# rates s and b for given
#  - true labels: y_true_label
#  - predicted labels: y_pred_label
#  - weight: weight
func_compute_s_b <- function(y_true_label, y_pred_label, weight){
  # From line [17] and line [38] in the provided starting kit as in
  # http://higgsml.lal.in2p3.fr/software/starting-kit/
  # there are the following important comments:
  # "don't forget to renormalize the weights to the same sum 
  # as in the complete training set" and implemented with the ratio var:
  # wFactor = 1.* numPoints / numPointsValidation
  # where numPoints and numPointsValidation are the number of the full
  # training set and the held-out validation set.
  #numTrain <- 250000.0
  #weight <- weight * (numTrain/length(weight))
  s <- sum(weight[y_true_label=='s' & y_pred_label=='s'])
  b <- sum(weight[y_true_label=='b' & y_pred_label=='s'])
  return(list(s, b))
}

#### This function computes approximate median significance (AMS) for given
# s and b, which are unnormalized true positive and false positive rates,
# respectively
func_compute_AMS <- function(s, b){
  bReg <- 10.0
  ams <- sqrt(2 * ((s + b + bReg) * log(1 + s / (b + bReg)) - s))
  return(ams)
}


#### This function trains a gbm classifier
trainGBMClassifier <- function(dfTrain, gbm_params, weights_opt, predictors, plot_on=FALSE){
  
  gc(reset=TRUE)
  
  # deal with weights for gbm
  if(weights_opt=='raw'){
    weights <- dfTrain$Weight
    weights <- weights/sum(weights)
  }else if(weights_opt=='balance'){
    # re-scale weights to have balanced weights
    weights <- dfTrain$Weight
    ind_pos <- which(dfTrain$Label_binary==1)
    ind_neg <- which(dfTrain$Label_binary==0)
    sum_wpos <- sum(weights[ind_pos])
    sum_wneg <- sum(weights[ind_neg])
    weights[ind_pos] <- 0.5 * (weights[ind_pos]/sum_wpos)
    weights[ind_neg] <- 0.5 * (weights[ind_neg]/sum_wneg)
  }else if(weights_opt=='none'){
    weights <- rep(1, dim(dfTrain)[1])
    weights <- weights/sum(weights)
  }
  
  # train gbm
  model <- gbm(Label_binary ~ .,
               data = dfTrain[, c(predictors, 'Label_binary')],
               distribution = gbm_params$distribution,
               weights = weights,
               n.trees = gbm_params$n.trees,
               shrinkage = gbm_params$shrinkage,
               interaction.depth = gbm_params$interaction.depth,
               n.minobsinnode = gbm_params$n.minobsinnode,
               train.fraction = gbm_params$train.fraction,
               bag.fraction = gbm_params$bag.fraction,
               cv.folds = gbm_params$cv.folds,
               class.stratify.cv = gbm_params$class.stratify.cv,
               verbose = gbm_params$verbose,
               n.cores = gbm_params$n.cores,
               keep.data = gbm_params$keep.data)
  
  # plot the error
  if(plot_on ==TRUE){
    if(gbm_params$cv.folds>1){
      best.iter <- gbm.perf(model, method="cv")
      min.cv.error <- min(model$cv.error)
      abline(h=min.cv.error, col='blue', lwd=2, lty=2)
    }    
  }
  return(model)
}


########################
## Training & Testing ##
########################

#### This function uses cv to choose the best cutoff
cvForParams <- function(dfTrain, gbm_params, weights_opt, predictors, cutoffs,
                        outer_cv_folds=5, inner_cv_folds=5, random_seed=2014){
  gc(reset=TRUE)
  ## random seed to ensure reproduciable results
  set.seed(random_seed)
  
  ## the params for the transformation, they are only used when transfom='logit'
  AMS <- matrix(0, outer_cv_folds, length(cutoffs))
  
  allIndex_outer <- seq(1, dim(dfTrain)[1])
  cvTrainIndex_outer <- createFolds(dfTrain$Label_binary, outer_cv_folds,
                                    list=TRUE, returnTrain=TRUE)
  cvTestIndex_outer <- lapply(cvTrainIndex_outer,
                              function(x)allIndex_outer[!allIndex_outer %in% x])
  
  for(outer_fold in seq(1, outer_cv_folds)){
    cat('\n=========================\n')
    cat('Perform ', inner_cv_folds,
        '-fold inner cross-validation on outer CV fold: ', outer_fold,
        '\n', sep='')
    ## this now contains 80% data for training, and 20% data for testing
    training <- dfTrain[cvTrainIndex_outer[[outer_fold]],]
    #     testing <- dfTrain[cvTestIndex_outer[[outer_fold]],]
    
    allIndex_inner <- seq(1, dim(training)[1])
    cvTrainIndex_inner <- createFolds(training$Label_binary, inner_cv_folds,
                                      list=TRUE, returnTrain=TRUE)
    cvTestIndex_inner <- lapply(cvTrainIndex_inner,
                                function(x)allIndex_inner[!allIndex_inner %in% x])
    #cat('Done\n')
    
    #prob <- rep(0, dim(training)[1])
    for(inner_fold in seq(1, inner_cv_folds)){
      
      trnInd <- cvTrainIndex_inner[[inner_fold]]
      tstInd <- cvTestIndex_inner[[inner_fold]]
      cat('\n-------------------------\n')
      cat('Inner CV fold: ', inner_fold, '\n', sep='')
      
      #### We first train defaulter classifier
      cat('Train GBM classifier.\n')
      model <- trainGBMClassifier(
        dfTrain = training[trnInd,],
        gbm_params = gbm_params,
        weights_opt = weights_opt,
        predictors = predictors)
      print(summary(model))
      cat('Done.\n')
      
      
      ####
      ## make prediction on the valid set
      if(gbm_params$cv.folds>1){
        best.iter <- gbm.perf(model, method="cv", plot.it=FALSE)
      }else{
        best.iter <- gbm_params$n.trees
      }
      prob <- predict(model, newdata=training[tstInd,],
                      n.trees=best.iter, type="response")
      
      auc <- func_compute_AUC(training$Label_binary[tstInd], prob, plotROC=TRUE)
      
      ## compute the f1-score for varying cutoffs
      cat('Compute AMS for varying cutoffs.\n')
      w_rescale <- sum(dfTrain$Weight)/sum(training$Weight[tstInd])
      this_AMS <- func_compute_AMS_cutoffs(training$Label[tstInd], prob,
                                           training$Weight[tstInd]*w_rescale, cutoffs)
      ## accumulate the f1-score for this iteration-fold
      AMS[outer_fold,] <- AMS[outer_fold,] + this_AMS
      cat('Max AMS for this inner cv fold: ', max(this_AMS),
          ' at cutoff: ', cutoffs[which.max(this_AMS)], '\n', sep='')
      cat('Done.\n')
      
    }
    
    cat('Done for outer CV fold: ', outer_fold, '\n', sep='')
  }
  
  # aveage over outer_cv_folds & cv_folds
  AMS <- AMS/(inner_cv_folds)
  # find the 
  best_AMS <- rep(0, outer_cv_folds)
  best_cutoff <- rep(0, outer_cv_folds)
  
  
  dir.create('./Figure', showWarnings=FALSE, recursive=TRUE)
  png('./Figure/AMS_cutoffs_Inner_CV.png')
  par(mfrow=c(outer_cv_folds, 1))
  for(outer_fold in seq(1, outer_cv_folds)){
    best_AMS[outer_fold] <- max(AMS[outer_fold,])
    best_cutoff[outer_fold] <- cutoffs[which.max(AMS[outer_fold,])]
    
    plot(cutoffs, AMS[outer_fold,], col='green', type='l', lwd=2,
         xlab='Cutoff', ylab='AMS',
         main=paste('AMS: ', round(best_AMS[outer_fold],5),
                    ' Cutoff: ', round(best_cutoff[outer_fold],5), sep=''))
    abline(v=best_cutoff[outer_fold], col='blue', lwd=2, lty=2)
  }
  dev.off()
  
  cat('\n-------- Summary --------\n\n', sep='')
  cat('Best cutoff on outer CV training set:', best_cutoff, '\n')
  cat('Best AMS on outer CV training set:', best_AMS, '\n')
  cat('\n-------------------------\n', sep='')
  ##
  return(best_cutoff)
}


#### This function evaluates the performance on the held out testing set
cvForPerformance <- function(dfTrain, gbm_params, weights_opt, predictors, cutoffs, 
                             best_cutoff, outer_cv_folds, random_seed=2014){
  
  gc(reset=TRUE)
  # random seed to ensure reproduciable results
  # we use the same random seed as that in the cv for best cutoff
  # in this case, the testing set is not used in the cv for best cutoff
  # and it is a hold out set that can be used to evaluate the performance
  set.seed(random_seed)
  
  # AMS estimated on the hold out testing set during each k-fold cv
  # this is used to estimate the CV performance on the CV testing set
  # with the best_cutoff estimated from the CV training set with inner CV
  AMS_CV <- rep(0, outer_cv_folds)
  # this is used to re-choose the best cutoff for training the final model
  AMS_cutoffs <- rep(0, length(cutoffs))
  
  
  allIndex_outer <- seq(1, dim(dfTrain)[1])
  cvTrainIndex_outer <- createFolds(dfTrain$Label_binary, outer_cv_folds,
                                    list=TRUE, returnTrain=TRUE)
  cvTestIndex_outer <- lapply(cvTrainIndex_outer,
                              function(x)allIndex_outer[!allIndex_outer %in% x])
  
  #prob <- rep(0, dim(dfTrain)[1])
  ## random seed to ensure reproduciable results
  for(outer_fold in seq(1, outer_cv_folds)){
    
    cat('\n=========================\n')
    cat('Outer CV fold: ', outer_fold, '\n', sep='')
    
    ## this now contains 80% data for training, and 20% data for testing
    trnInd <- cvTrainIndex_outer[[outer_fold]]
    tstInd <- cvTestIndex_outer[[outer_fold]]
    
    #### We first train defaulter classifier
    cat('Train GBM classifier.\n')
    model <- trainGBMClassifier(
      dfTrain = dfTrain[trnInd,],
      gbm_params = gbm_params,
      weights_opt = weights_opt,
      predictors = predictors)
    cat('Done.\n')
    
    #### compute MAE on the testing set
    
    ## make prediction on the valid set
    if(gbm_params$cv.folds>1){
      best.iter <- gbm.perf(model, method="cv", plot.it=FALSE)
    }else{
      best.iter <- gbm_params$n.trees
    }
    prob <- predict(model, newdata=dfTrain[tstInd,],
                    n.trees=best.iter, type="response")
    
    auc <- func_compute_AUC(dfTrain$Label_binary[tstInd], prob, plotROC=TRUE)
    
    cat('Compute AMS with best cutoff.\n')
    w_rescale <- sum(dfTrain$Weight)/sum(dfTrain$Weight[tstInd])
    this_ams <- func_compute_AMS_cutoffs(dfTrain$Label[tstInd], prob,
                                         dfTrain$Weight[tstInd]*w_rescale,
                                         best_cutoff[outer_fold])
    AMS_CV[outer_fold] <- this_ams
    cat('Done.\n')
    cat('Compute AMS for varying cutoffs.\n')
    w_rescale <- sum(dfTrain$Weight)/sum(dfTrain$Weight[tstInd])
    this_ams <- func_compute_AMS_cutoffs(dfTrain$Label[tstInd], prob,
                                         dfTrain$Weight[tstInd]*w_rescale, cutoffs)
    AMS_cutoffs  <- AMS_cutoffs  + this_ams
    cat('Done.\n')
  }
  #AMS_CV <- AMS_CV/outer_cv_folds
  AMS_cutoffs <- AMS_cutoffs/outer_cv_folds
  # find the 
  best_AMS_final <- max(AMS_cutoffs)
  best_cutoff_final <- cutoffs[which.max(AMS_cutoffs)]
  png('./Figure/AMS_cutoffs_Outer_CV.png')
  plot(cutoffs, AMS_cutoffs, col='green', type='l', lwd=2,
       xlab='Cutoff', ylab='AMS',
       main=paste('AMS: ', round(best_AMS_final,5),
                  ' Cutoff: ', round(best_cutoff_final,5), sep=''))
  abline(v=best_cutoff_final, col='blue', lwd=2, lty=2)
  dev.off()
  cat('\n-------- Summary --------\n\n', sep='')
  cat('AMS for ', outer_fold,
      '-fold outer cross-validation: Mean = ', round(mean(AMS_CV), 5),
      ' SD = ', round(sd(AMS_CV), 5), '\n\n', sep='')
  cat('Final best cutoff: ', best_cutoff_final, '\n',
      'Final best AMS: ', best_AMS_final, '\n', sep='')
  cat('\n-------------------------\n', sep='')
  
  return(list(AMS_CV, best_cutoff_final))
}


#### This function makes the final submission
makeSubmission <- function(dfTrain, dfTest, gbm_params, weights_opt, predictors,
                           best_cutoff, random_seed=2014, saveFileName){
  
  gc(reset=TRUE)
  # random seed to ensure reproduciable results
  set.seed(random_seed)
  
  #### We first train gbm classifier
  cat('Train GBM classifier.\n')
  model <- trainGBMClassifier(
    dfTrain = dfTrain,
    gbm_params = gbm_params,
    weights_opt = weights_opt,
    predictors = predictors)
  
  ## compute the probability of default as additional feature
  if(gbm_params$cv.folds>1){
    best.iter <- gbm.perf(model, method="cv", plot.it=FALSE)
  }else{
    best.iter <- gbm_params$n.trees
  }
  prob_test <- predict(model, newdata=dfTest,  n.trees=best.iter, type="response")
  
  cat('Done.\n')
  prob_train <- predict(model, newdata=dfTrain,  n.trees=best.iter, type="response")
  
  auc <- func_compute_AUC(dfTrain$Label_binary, prob_train, plotROC=TRUE)
  
  AMS_train <- func_compute_AMS_cutoffs(dfTrain$Label, prob_train, dfTrain$Weight, best_cutoff)
  cat('\nAMS for the WHOLE training set with best_cutoff: AMS = ', AMS_train, '\n', sep='')
  
  
  RankOrder <- rank(prob_test, ties.method='random')
  top <- as.integer(floor(best_cutoff * length(prob_test)))
  thresh <- prob_test[which(RankOrder==top)]
  Label_pred <- ifelse(prob_test>thresh, 's', 'b')
  
  sub <- data.frame(EventId=dfTest$EventId, RankOrder=RankOrder, Class=Label_pred)
  
  ## make submission
  write.csv(sub, saveFileName, row.names=F, quote=F)
  
  return(model)
}


##########
## Main ##
##########

# read in data
dfTrain <- as.data.frame(fread('./Data/training.csv', header=T))
dfTrain[dfTrain==-999.0] <- NA # no need for this
dfTrain$Label_binary <- ifelse(dfTrain$Label=='s', 1, 0)
dfTest <- as.data.frame(fread('./Data/test.csv', header=T))
dfTest[dfTest==-999.0] <- NA # no need for this

# convert to factor
dfTrain$PRI_jet_num <- as.factor(dfTrain$PRI_jet_num)
dfTest$PRI_jet_num <- as.factor(dfTest$PRI_jet_num)

# names for predictors
predictors <- colnames(dfTrain)
NotPredictors <- c('EventId', 'Weight', 'Label', 'Label_binary')
predictors <- predictors[-which(predictors %in% NotPredictors)]


# path to save RData data
filePath_RData <- './Submission/GBM/CV/RData'
dir.create(filePath_RData, showWarnings=FALSE, recursive=TRUE)

# path to save csv submission
filePath_csv <- './Submission/GBM/CV/csv'
dir.create(filePath_csv, showWarnings=FALSE, recursive=TRUE)


#### setup for the cv
# k-fold cv
inner_cv_folds <- 2
# times of performing k-fold cv
outer_cv_folds <- 2
# number of random seed
seed_num <- 1
# random seeds
set.seed(2014) # to ensure reproducable results
random_seeds <- sample(10000, seed_num)

# the estimated cv MAE for each seed and cv
AMS_CV <- matrix(0, seed_num, outer_cv_folds)

# varying cutoffs
step <- 0.01
cutoffs <- seq(0.0, 1.0, step)


weights_opts <- c('none', 'balance', 'raw')[1]

# parameters for GBM model
gbm_params <- list(distribution = 'bernoulli',
                   n.trees = 100,
                   shrinkage = 0.1,
                   interaction.depth = 10,
                   n.minobsinnode = 10,
                   train.fraction = 1.0,
                   bag.fraction = 0.5,
                   cv.folds = 2,
                   n.cores = 2,
                   class.stratify.cv = TRUE,
                   verbose = TRUE,
                   keep.data = FALSE)

# acutal training code
for(weights_opt in weights_opts){
  for(count_seed in seq(1, length(random_seeds))){
    
    seed <- random_seeds[count_seed]
    gc(reset=T)
    
    cat('The ', count_seed, ' seed ...', '\n', sep='')
    
    
    #### cross-validate to find the best cutoff
    cat('Cross-validation for the best cutoff ...\n', sep='')
    best_cutoff <- cvForParams(
      dfTrain, gbm_params, weights_opt, predictors,
      cutoffs, outer_cv_folds, inner_cv_folds, seed
    )
    
    
    #### cross-validate to estimate MAE using the found best cutoff
    cat('Cross-validation for estimating the AMS ...\n', sep='')
    results <- cvForPerformance(
      dfTrain, gbm_params, weights_opt, predictors,
      cutoffs, best_cutoff, outer_cv_folds, seed
    )
    AMS_CV[count_seed,] <- results[[1]]
    best_cutoff_final <- results[[2]]
    
    
    saveFileName <- paste(filePath_RData, '/GBM_',
                          '[Weight_', capitalize(weights_opt),']_', 
                          '[', capitalize(gbm_params$distribution),']_',
                          '[Ntree',gbm_params$n.trees,']_',
                          '[lr', gbm_params$shrinkage,']_',
                          '[Bag', gbm_params$bag.fraction,']_',
                          '[Cutoff', best_cutoff_final, ']_',
                          '[AMS', round(mean(AMS_CV[count_seed,]),5), ']_',
                          '[SD', round(sd(AMS_CV[count_seed,]),5), ']_',
                          '[Seed', seed,']',
                          '.RData', sep='')
    save(list=c('AMS_CV','best_cutoff_final'), file=saveFileName)
    
    
    #### retrain the model and make a submission
    cat('Make submission ...\n', sep='')
    saveFileName <- paste(filePath_csv, '/GBM_',
                          '[Weight_', capitalize(weights_opt),']_', 
                          '[', capitalize(gbm_params$distribution),']_',
                          '[Ntree',gbm_params$n.trees,']_',
                          '[lr', gbm_params$shrinkage,']_',
                          '[Bag', gbm_params$bag.fraction,']_',
                          '[Cutoff', best_cutoff_final, ']_',
                          '[AMS', round(mean(AMS_CV[count_seed,]),5), ']_',
                          '[SD', round(sd(AMS_CV[count_seed,]),5), ']_',
                          '[Seed', seed,']',
                          '.csv', sep='')
    
    # If you don't want to perform CV, just set best_cutoff_final, e.g., 0.85 and run 
    # makeSubmission function
    model <- makeSubmission(dfTrain, dfTest, gbm_params, weights_opt, predictors,
                            best_cutoff_final, seed, saveFileName)
    
  }
}

