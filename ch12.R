library("AppliedPredictiveModeling")
library('subselect')
library('rms')
library('MASS')
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
