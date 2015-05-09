library("AppliedPredictiveModeling")
library('caret')
library('C50')
library('gbm')
library('ipred')
library('partykit')
library('pROC')
library('randomForest')
library('RWeka')
library('rpart')
library('e1071')

# note - the book's computation section appears to be out of date. Work
# from here instead: https://github.com/cran/AppliedPredictiveModeling/blob/master/inst/chapters/14_Class_Trees.R

# Classification Trees

ctrl <- trainControl(method = "LOGCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)


set.seed(476)
rpartFit <- train(x = training[, fullSet],
                  y = training$Class,
                  method = 'rpart',
                  tuneLength = 30,
                  metric = "ROC",
                  trControl = ctrl)

plot(as.party(rpartFit$finalModel))

rpart2008 <- merge(rpartFit$pred, rpartFit$bestTune)
rpartCM <- confusionMatrix(rpartFit, norm = "none")
rpartCM

rpartRoc <- roc(response = rpartFit$pred$obs,
                predictor = rpartFit$pred$successful,
                levels = rev(levels(rpartFit$pred$obs)))
rpartRoc
plot(rpartRoc)

rpartFactorFit <- train(x = training[, factorPredictors],
                        y = training$Class,
                        method = "rpart",
                        tuneLength = 30,
                        metric = "ROC",
                        trControl = ctrl)

rpartFactorFit
plot(as.party(rpartFactorFit$finalModel))

rpartFactor2008 <- merge(rpartFactorFit$pred, rpartFactorFit$bestTune)
rpartFactorCM <- confusionMatrix(rpartFactorFit, norm = 'none')
rpartFactorCM

rpartFactorRoc <- roc(response = rpartFactorFit$pred$obs,
                      predictor = rpartFactorFit$pred$successful,
                      levels = rev(levels(rpartFactorFit$pred$obs)))
plot(rpartFactorRoc)

# omnibus plot
plot(
  rpartRoc, type='s', print.thres = c(0.5),
  print.thres.pch = 3,
  print.thres.pattern = "",
  print.thres.cex = 1.2,
  col = 'red', legacy.axes = TRUE,
  print.thres.col = 'red')

plot(
  rpartFactorRoc, type='s', print.thres = c(0.5),
  print.thres.pch = 16,
  print.thres.pattern = "",
  print.thres.cex = 1.2,
  add = TRUE,
  col = 'black', legacy.axes = TRUE,
  print.thres.col = 'red')
legend(.75, .2,
       c('Grouped Categories', 'Independent Categories'),
       lwd = c(1,1),
       col = c('black', 'red'),
       pch = c(16, 3))

# woot! Let's do J48

j48FactorFit <- train(x = training[, factorPredictors],
                        y = training$Class,
                        method = "J48",
                        metric = "ROC",
                        trControl = ctrl)

j48FactorFit
# rule based models
# ugggg --- java hell. Come back to this one

# Bagged Trees
treeBagFit <- train(x = training[, fullSet],
                  y = training$Class,
                        nbagg = 50,
                        method = "treebag",
                        metric = "ROC",
                        trControl = ctrl)
treebag2008 <- merge(treeBagFit$pred,  treeBagFit$bestTune)
treebagCM <- confusionMatrix(treeBagFit, norm = "none")
treebagCM

treebagRoc <- roc(response = treeBagFit$pred$obs,
                  predictor = treeBagFit$pred$successful,
                  levels = rev(levels(treeBagFit$pred$obs)))
