# Ch 16 - remedies for severe class imbalance
library('AppliedPredictiveModeling')
library('caret')
library('C50')
library('DMwR')
library('DWD')
library('kernlab')
library('pROC')
library('rpart')
library('earth')

# note: DWD needs to be downloaded from the CRAN archive as it was recently removed
# from CRAN.

data(ticdata)

# recode some of the vars
View(ticdata)

isOrdered <- unlist(lapply(ticdata, function(x) any(class(x) == "ordered")))
recodeLevels <- function(x) {
  x <- gsub("f ", "", as.character(x))
  x <- gsub(" - ", "_to_", x)
  x <- gsub("-", "_to_", x)
  x <- gsub("%", "", x)
  x <- gsub("?", "Unk", x, fixed = TRUE)
  x <- gsub("[,'\\(\\)]", "", x)
  x <- gsub(" ", "_", x)
  factor(paste("_", x, sep = ""))
}

convertCols <- c("STYPE", "MGEMLEEF", "MOSHOOFD",
                 names(isOrdered)[isOrdered])

for(i in convertCols) {
  ticdata[, i] <- factor(gsub(" ", "0",format(as.numeric(ticdata[,i]))))
}

ticdata$CARAVAN <- factor(as.character(ticdata$CARAVAN),
                          levels = rev(levels(ticdata$CARAVAN)))

set.seed(156)
split1 <- createDataPartition(ticdata$CARAVAN, p = 0.7)[[1]]
other <- ticdata[-split1,]
training <- ticdata[split1,]

set.seed(934)
split2 <- createDataPartition(other$CARAVAN, p = 1/3)[[1]]
evaluation <- other[-split2,]
testing <- other[split2,]

predictors <- names(training)[names(training) != "CARAVAN"]
predictors

trainingInd <- data.frame(model.matrix(CARAVAN ~ . ,
                                       data = training))[, -1]

evaluationInd <- data.frame(model.matrix(CARAVAN ~ . ,
                                       data = evaluation))[, -1]

testingInd <- data.frame(model.matrix(CARAVAN ~ . ,
                                       data = testing))[, -1]

trainingInd$CARAVAN <- training$CARAVAN
evaluationInd$CARAVAN <- evaluation$CARAVAN
testingInd$CARAVAN <- testing$CARAVAN

# NZV = near zero variance
isNZV <- nearZeroVar(trainingInd)
noNZVSet <- names(trainingInd)[-isNZV]
noNZVSet
isNZV

testResults <- data.frame(CARAVAN = testing$CARAVAN)
evalResults <- data.frame(CARAVAN = evaluation$CARAVAN)

fiveStats <- function(...) {
  c(twoClassSummary(...),
    defaultSummary(...))
}

fourStats <- function(data, lev=levels(data$obs), model = NULL) {
  accKapp <- postResample(data[, 'pred'], data[, 'obs'])
  out <- c(accKapp,
           sensitivity(data[, 'pred'], data[, 'obs'], lev[1]),
           sensitivity(data[, 'pred'], data[, 'obs'], lev[2])
           )
  names(out)[3:4] <- c("Sens", "Spec")
  out
}

ctrl <- trainControl(method = 'cv',
                     classProbs = TRUE,
                     summaryFunction = fiveStats,
                     verboseIter = TRUE)
ctrlNoProb <- ctrl
ctrlNoProb$summaryFunction <- fourStats
ctrlNoProb$classProbs <- FALSE

cores <- 8

if(cores > 1) {
  library(doMC)
  registerDoMC(cores - 1)
}

# df to hold results:
evalResults <- data.frame(CARAVAN = evaluation$CARAVAN)

set.seed(1401) #magic?
rfFit <- train(CARAVAN ~ .,
               data = trainingInd,
               method = 'rf',
               trControl = ctrl,
               ntree = 1500,
               tuneLength = 5,
               metric = "ROC"
               )

lrFit <- train(CARAVAN ~ .,
               data = trainingInd[, noNZVSet],
               method = 'glm',
               trControl = ctrl,
               metric = "ROC"
               )
lrEvalPred <- predict(lrFit, 
                      evaluationInd[, noNZVSet], 
                      type = "prob")

lrTestPred <- predict(lrFit, 
                      testingInd[, noNZVSet],
                      type = "prob")

evalResults$LogReg <- lrEvalPred[, 1]
testResults$LogReg <- lrTestPred[,1]

lrRoc <- roc(evalResults$CARAVAN, evalResults$LogReg,
             levels = rev(levels(evalResults$CARAVAN)))
plot(lrRoc, legacy.axes = TRUE)

fdaFit <- train(CARAVAN ~ .,
               data = training,
               method = 'fda',
               tuneGrid = data.frame(.degree = 1, .nprune = 1:25),
               trControl = ctrl,
               metric = "ROC"
               )

fdaEvalPred <- predict(fdaFit,
                      newdata = evaluation[, predictors],
                      type = 'prob')

fdaTestPred <- predict(fdaFit,
                      newdata = testing[, predictors],
                      type = 'prob')

evalResults$FDA <- fdaEvalPred[, 1]
testResults$FDA <- fdaTestPred[,1]


fdaRoc <- roc(evalResults$CARAVAN, evalResults$FDA,
             levels = rev(levels(evalResults$CARAVAN)))
plot(fdaRoc, legacy.axes = TRUE)

# plot all models on the same graph:
plot(fdaRoc, legacy.axes = TRUE, type="S", col='green')
plot(lrRoc, legacy.axes = TRUE, type="S", col='blue', add=TRUE)

labs <- c( #RF = "Random Forest",
          LogReg = "Logistic Regression",
          FDA = "FDA (MARS)")
lift1 <- lift(CARAVAN ~ LogReg + FDA,
              data = evalResults,
              labels = labs)
lift1

xyplot(lift1,
       ylab = "%Events Found",
       xlab = "% Customers Eval'd",
       lwd = 2,
       type = "l")

# Alternate Cutoffs
# TODO come back to this after running the rf mode

# Adjusting Priors

priors <- table(ticdata$CARAVAN)/nrow(ticdata) * 100
priors
fdaPriors <- fdaFit
fdaPriors$finalModel$prior <- c(insurance = 0.6, noinsurance = 0.4)

fdaPriorPred <- predict(fdaPriors, evaluation[, predictors])

fdaPriorProb <- predict(fdaPriors,
                        newdata = evaluation[, predictors],
                        type = 'prob')
evalResults$FDAPrior <- fdaPriorProb[, 1]


fdaPriorProbTest <- predict(fdaPriors,
                        newdata = testing[, predictors],
                        type = 'prob')
testResults$FDAPrior <- fdaPriorProbTest[, 1]

# confusion matix:
fdaPriorCM <- confusionMatrix(fdaPriorPred, evaluation$CARAVAN)
fdaPriorCM

fdaPriorROC <- roc(testResults$CARAVAN,
                   testResults$FDAPrior,
                   levels = rev(levels(testResults$CARAVAN)))
fdaPriorROC
plot(fdaPriorROC, legacy.axes = TRUE)

## How does it compare to regular FDA?

plot(fdaPriorROC, legacy.axes = TRUE, col="green")
plot(fdaRoc, legacy.axes = TRUE, col="blue", add = TRUE)

# Sampling
## TODO come back to resampling for RF

# Cost Sensitive Training:

sigma <- sigest(CARAVAN ~ .,
                data = trainingInd[, noNZVSet],
                frac = 0.75)
names(sigma) <- NULL

svmGrid1 <- data.frame(sigma = sigma[2],
                       C = 2^c(2:10))
svmGrid1

svmFit <- train(CARAVAN ~ .,
                data = trainingInd[, noNZVSet],
                method = "svmRadial",
                tuneGrid = svmGrid1,
                preProc = c('center', 'scale'),
                metric = 'Kappa',
                trControl = ctrl)
svmFit

svmEvalPred <- predict(svmFit, 
                       newdata = evaluationInd[, noNZVSet],
                       type = 'prob')

evalResults$SVM <- svmEvalPred[, 1]

svmTestPred <- predict(svmFit,
                       newdata = testingInd[, noNZVSet],
                       type = 'prob')

testResults$SVM <- svmTestPred[, 1]

svmEvalRoc <- roc(evalResults$CARAVAN,
                  evalResults$SVM,
                  levels = rev(levels(evalResults$CARAVAN)))
svmEvalRoc

svmTestRoc <- roc(testResults$CARAVAN,
                  testResults$SVM,
                  levels = rev(levels(testResults$CARAVAN)))
svmTestRoc

svmEvalFitClass <- predict(svmFit,
                           evaluationInd[, noNZVSet])
confusionMatrix(svmEvalFitClass, evalResults$CARAVAN)

svmTestFitClass <- predict(svmFit,
                           testingInd[, noNZVSet])
confusionMatrix(svmTestFitClass, testResults$CARAVAN)

set.seed(1401)
svmWtFit <- train(CARAVAN ~ .,
                  data = trainingInd[, noNZVSet],
                  method = "svmRadial",
                  tuneGrid = svmGrid1,
                  preProc = c("center", "scale"),
                  metric = "Kappa",
                  class.weights = c(insurance = 18, noinsurance = 1),
                  trControl = ctrl)
svmWtFit

svmWtFitEvalPred <- predict(svmWtFit, newdata = evaluationInd[, noNZVSet])
confusionMatrix(svmWtFitEvalPred, evalResults$CARAVAN)

svmWtFitTestPred <- predict(svmWtFit, newdata = testingInd[, noNZVSet])
confusionMatrix(svmWtFitTestPred, testResults$CARAVAN)

initialRpart <- rpart(CARAVAN ~ ., data = training,
                      control = rpart.control(cp = 0.0001))
rpartGrid <- data.frame(cp = initialRpart$cptable[, "CP"])
rpartGrid

cmat <- list(loss = matrix(c(0, 1, 20, 0), ncol = 2))
set.seed(1401)
cartWMod <- train(x = training[,predictors],
                  y = training$CARAVAN,
                  method = "rpart",
                  trControl = ctrlNoProb,
                  tuneGrid = rpartGrid,
                  metric = "Kappa",
                  parms = cmat)
