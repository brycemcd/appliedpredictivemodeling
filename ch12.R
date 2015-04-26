library("AppliedPredictiveModeling")
library('subselect')
library('rms')
library('MASS')
library('pls')
library('glmnet')
# creates data set to play around with. Copied/Pasted from
# https://raw.githubusercontent.com/cran/AppliedPredictiveModeling/master/inst/chapters/CreateGrantData.R

# important! Download unimelb_training.csv from https://www.kaggle.com/c/unimelb/data?unimelb_example.csv
source('createGrantData.R')

length(fullSet)
length(reducedSet)

head(fullSet)
head(reducedSet)

reducedCovMat <- cov(training[, reducedSet])
trimmingResults <- trim.matrix(reducedCovMat)

# no vars discarded:
trimmingResults$names.discarded

fullCovMat <- cov(training[, fullSet])
fullSetResults <- trim.matrix(fullCovMat)

# collinear vars discarded:
fullSetResults$names.discarded

modelFit <- glm(Class ~ Day,
                data = training[pre2008, ],
                family = 'binomial')
modelFit
successProb <- 1 - predict(modelFit,
                           newdata = data.frame(Day = c(10, 150, 300, 350)),
                           type = 'response')
successProb

rcsFit <- lrm(Class ~ rcs(Day),
              data= training[pre2008, ])
rcsFit
# p values in Day, Day', Day'' and Day''' all indicate a nonlinear relationship is probable

opp <- function(x) {
  -x
}
dayProfile <- Predict(rcsFit,
                      Day = 0:365,
                      fun = 'opp')
plot(dayProfile, ylab='Log Odds')


# using caret
ctrl <- trainControl( method = 'LGOCV',
                      summaryFunction = twoClassSummary,
                      classProbs = TRUE,
                      index = list(TrainSet = pre2008),
                      savePredictions = TRUE)

set.seed(476)
lrFull <- train(training[, fullSet],
                y= training$Class,
                method = 'glm',
                metric = 'ROC',
                trControl = ctrl)

lrReduced <- train(training[, reducedSet],
                   y = training$Class,
                   method = 'glm',
                   metric = "ROC",
                   trControl = ctrl)

head(lrReduced$pred)

confusionMatrix(data = lrReduced$pred$pred,
                reference = lrReduced$pred$obs)

reducedRoc <- roc(response = lrReduced$pred$obs,
                  predictor = lrReduced$pred$successful,
                  levels = rev(levels(lrReduced$pred$obs)))
plot(reducedRoc, legacy.axes = TRUE)
auc(reducedRoc)

#LDA
grantPreProcess <- preProcess(training[pre2008, reducedSet])
grantPreProcess

scaledPre2008 <- predict(grantPreProcess, newdata = training[pre2008, reducedSet])
scaled2008HoldOut <- predict(grantPreProcess, newdata = training[-pre2008, reducedSet])

ldaModel <- lda(x = scaledPre2008,
                grouping = training$Class[pre2008])

head(ldaModel$scaling)
plot(ldaModel)

ldaHoldOutPredictions <- predict(ldaModel, scaled2008HoldOut)
head(ldaHoldOutPredictions$class)
head(ldaHoldOutPredictions$posterior)

# now with more caret
set.seed(476)
ldaFit1 <- train(x = training[, reducedSet],
                 y = training$Class,
                 method =  'lda',
                 preProc = c('center', 'scale'),
                 metric = 'ROC',
                 trControl = ctrl)
ldaFit1

ldaTestClasses <- predict(ldaFit1,
                          newdata = testing[, reducedSet])
ldaTestProbs <- predict(ldaFit1,
                        newdata = testing[, reducedSet],
                        type = 'prob')
head(ldaTestProbs)

# PLS discriminant Analysis

plsFit2 <- train(x = training[, reducedSet],
                 y = training$Class,
                 method = 'pls',
                 tuneGrid = expand.grid(.ncomp = 1:10),
                 preProc = c('center', 'scale'),
                 metric = 'ROC',
                 trControl = ctrl)
plsProbs <- predict(plsFit2, newdata = training[-2008, reducedSet],
                    type = 'prob')
head(plsProbs)
plsImpGrant <- varImp(plsFit2, scale = FALSE)
plsImpGrant
plot(plsImpGrant, top=20, scales = list(y = list(cex = 0.95)))

# Penalized Models

glmnetModel <- glmnet(x = as.matrix(training[,fullSet]),
                      y = training$Class,
                      family = 'binomial')
predict(glmnetModel,
        newx = as.matrix(training[1:5, fullSet]),
        s = c(0.05, 0.1, 0.2),
        type = 'class')

glmnGrid <- expand.grid(.alpha = c(0, 0.1, 0.2, 0.4, 0.6, 0.8, 1),
                        .lambda = seq(0.01, 0.2, length=40))
glmnGrid
glmnTuned <- train(training[, fullSet],
                   y = training$Class,
                   method = 'glmnet',
                   tuneGrid = glmnGrid,
                   preProc = c('center', 'scale'),
                   metric = 'ROC',
                   trControl = ctrl)

# nearest shrunken centroids
nscGrid <- data.frame(.threshold = 0:25)
nscTuned <- train(x = training[,fullSet],
                  y = training$Class,
                  method = 'pam',
                  preProc = c('center', 'scale'),
                  tuneGrid = nscGrid,
                  metric = 'ROC',
                  trControl = ctrl)

predictors(nscTuned)
varImp(nscTuned, scale = FALSE)

confusionMatrix(data = nscTuned$pred$pred,
                reference = nscTuned$pred$obs)

reducedRoc <- roc(response = nscTuned$pred$obs,
                  predictor = nscTuned$pred$successful,
                  levels = rev(levels(nscTuned$pred$obs)))
plot(reducedRoc, legacy.axes = TRUE)
