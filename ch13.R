# CH 13
library("AppliedPredictiveModeling")
library("caret")
library("earth")
library("kernlab")
library("klaR")
library("MASS")
library('mda')
library('nnet')
library('rrcov')
library('pROC')

source('createGrantData.R')

# NDA
source('linear-nonlinear-classification-ctrl.R')

set.seed(476) # magic #, not significant
mdaFit <- train(training[,reducedSet], training$Class,
                method = 'mda',
                metric = 'ROC',
                tuneGrid = expand.grid(.subclasses = 1:8),
                trControl = ctrl)

predict(mdaFit, newdata = head(training[-pre2008, reducedSet]))

# neural nets for classification

nnetMod <- nnet(Class ~ NumCI + CI.1960, data = training[pre2008, ],
                size = 3, decay = 0.1)
nnetMod
# class probs
predict(nnetMod, newdata = head(testing))
# classes
predict(nnetMod, newdata = head(testing), type='class')


# using caret
nnetGrid <- expand.grid(.size = 1:10,
                        .decay = c(0,0.1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- 1 * (maxSize * (length(reducedSet) + 1) + maxSize + 1)
nnetFit <- train(x = training[, reducedSet],
                 y = training$Class,
                 method = 'nnet',
                 metric = "ROC",
                 preProc = c("center", "scale", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl
                 )
predict(nnetFit, newdata = head(testing[, reducedSet]))

# SVM for classification

sigmaRangeReduced <- sigest(as.matrix(training[, reducedSet]))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2 ^(seq(-4, 4)))
smvRModel <- train(training[, reducedSet], training$Class,
                   method = 'svmRadial',
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   # other kernal functions can be defined via the kernel and kpar args
                   trControl = ctrl)
smvRModel

# classes
predict(smvRModel, newdata = head(training[-pre2008, reducedSet]))
# probs
predict(smvRModel, newdata = head(training[-pre2008, reducedSet]), type = 'prob')

# KNN

knnFit <- train(training[, reducedSet], training$Class,
                method = 'knn',
                metric = "ROC",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(.k = c(4*(0:5)+1,
                                             20*(1:5)+1,
                                             50*(2:9)+1)),
                trControl = ctrl
                )
knnFit
knnFit$pred <- merge(knnFit$pred, knnFit$bestTune)

knnRoc <- roc(response = knnFit$pred$obs,
              predictor = knnFit$pred$successful,
              levels = rev(levels(knnFit$pred$obs)))
plot(knnRoc, legacy.axes = TRUE)

# Naive Bayes

# need to massage the data a bit:
factors <- c("SponsorCode", "ContractValueBand", "Month", "Weekday")
nbPredictors <- factorPredictors[factorPredictors %in% reducedSet]
nbPredictors <- c(nbPredictors, factors)

nbTraining <- training[, c("Class", nbPredictors)]
nbTesting <- testing[, c("Class", nbPredictors)]

# shouldn't we use apply() here?
for(i in nbPredictors) {
  varLevels <- sort(unique(training[,i]))
  if(length(varLevels) <= 15) {
    nbTraining[, i] <- factor(nbTraining[, i],
                              levels = paste(varLevels))
    nbTesting[, i] <- factor(nbTraining[, i],
                              levels = paste(varLevels))
  }
}

nBayesFit <- NaiveBayes(Class ~ .,
                        data = nbTraining[pre2008,],
                        usekernel = TRUE,
                        fL = 2)
predict(nBayesFit, newdata = head(nbTesting))