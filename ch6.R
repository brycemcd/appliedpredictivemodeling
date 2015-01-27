# ch 6
library('AppliedPredictiveModeling')
library('caret')
library('MASS')
library('doMC')
library('pls')
library('elasticnet')
data(solubility)

set.seed(2)

trainingData <- solTrainXtrans
trainingData$Solubility <- solTrainY

lmFitAllPredictors <- lm(Solubility~., data=trainingData)
summary(lmFitAllPredictors)

# make prediction
lmPred1 <- predict(lmFitAllPredictors, solTestXtrans)
head(lmPred1)

lmValues1 <- data.frame(obs=solTestY, pred=lmPred1)
defaultSummary(lmValues1)
# ordinary least squares somewhat optimisitic for this set, try the Huber Approach:

rlmFitAllPredictors <- rlm(Solubility ~ ., data=trainingData)

rlmPred1 <- predict(rlmFitAllPredictors, solTestXtrans)
head(lmPred1)
rlmValues1 <- data.frame(obs=solTestY, pred=rlmPred1)
defaultSummary(rlmValues1)
# RMSE + R^2 slightly worse

# It's possible to use up the cores here:

registerDoMC(4)
ctrl <- trainControl(method='cv', number=10)
set.seed(10)
lmFit1 <- train(x=solTrainXtrans, y=solTrainY,
                method='lm', trControl=ctrl)
lmFit1

# test to see how well the predictions fit:
xyplot(solTrainY ~ predict(lmFit1),
       type=c('p', 'g'),
       xlab="Predicted", ylab="Observed")

xyplot(resid(lmFit1) ~ predict(lmFit1),
       type=c('p', 'g'),
       xlab="Predicted", ylab="Residuals")
# The tests show an ideal case. The observed values follow a line and the
# Residuals are a random cloud of points

# Reduce dimensions by removing highly correlated variables:
corThresh <- 0.9
tooHigh <- findCorrelation(cor(solTrainXtrans), corThresh)
corrPred <- names(solTrainXtrans)[tooHigh]
trainXfiltered <- solTrainXtrans[, -tooHigh]

testXfiltered <- solTestXtrans[, -tooHigh]
set.seed(100)
lmFiltered <- train(solTrainXtrans, solTrainY, method='lm', trControl=ctrl)
lmFiltered
# we get approximately the same results as before, but are sing ~ 30 fewer dimensions

# PCA can be applied prior to the regression:
rlmPCA <- train(solTrainXtrans, solTrainY,
                method='rlm',
                preProcess='pca',
                trControl=ctrl)
rlmPCA


# Play around with Partial Least Squares:
plsTune <- train(solTrainXtrans, solTrainY,
                 method='pls',
                 tuneLength=20,
                 trControl=ctrl,
                 preProc=c('center', 'scale'))
plsTune

pcrTune <- train(x = solTrainXtrans, y = solTrainY,
                 method = "pcr",
                 tuneGrid = expand.grid(ncomp = 1:35),
                 trControl = ctrl)
pcrTune

# compare models:
plsResamples <- plsTune$results
plsResamples$Model <- 'PLS'
pcrResamples <- pcrTune$results
pcrResamples$Model <- 'PCR'
plsPlotData <- rbind(plsResamples, pcrResamples)

# This is good stuff:
xyplot(RMSE ~ ncomp,
       data=plsPlotData,
       xlab="# components",
       ylab='RMSE (cv)',
       auto.key = list(columns=2),
       groups=Model,
       type= c('o', 'g'))

plsImp <- varImp(plsTune, scale = FALSE)
plot(plsImp, top = 25, scales = list(y = list(cex = .95)))

# Penalized Models:

# Ridge Regression
ridgeModel <- enet(x=as.matrix(solTrainXtrans), y=solTrainY, lambda=0.001)
ridgePred <- predict(ridgeModel, newx = as.matrix(solTestXtrans), s=1, model='fraction', type='fit')
head(ridgePred$fit)

ridgeGrid <- data.frame(.lambda = seq(0, 0.1, length=15))
ridgeRegFit <- train(solTrainXtrans, solTrainY,
                     method='ridge',
                     tuneGrid=ridgeGrid,
                     trControl=ctrl,
                     preProc=c('center', 'scale'))
ridgeRegFit

# Lasso:
enetModel <- enet(x=as.matrix(solTrainXtrans), y=solTrainY, lambda=0.01, normalize=TRUE)
enetPred <- predict(enetModel, newx=as.matrix(solTestXtrans), 
                    s=0.1, 
                    mode='fraction',
                    type='coefficients')
head(enetPred$coefficients)

enetGrid <- expand.grid(lambda = c(0, 0.01, .1), 
                        fraction = seq(.05, 1, length = 20))
enetTune <- train(solTrainXtrans, solTrainY,
                  method='enet',
                  tuneGrid=enetGrid,
                  trControl=ctrl,
                  preProc = c('center', 'scale'))
plot(enetTune)
