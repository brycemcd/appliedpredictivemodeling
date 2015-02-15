# Ch 10 (Ch9 was a summary chapter with no computing sections)
# Model Determination Case Study

library(AppliedPredictiveModeling)
library(caret)
library(plyr)

data(concrete)
str(concrete)
str(mixtures)

featurePlot(x = concrete[, -9],
            y = concrete$CompressiveStrength,
            between = list(x=1, y=1),
            type = c('g', 'p', 'smooth'))

averaged <- ddply(mixtures,
                  .(Cement, BlastFurnaceSlag, FlyAsh, Water,
                    Superplasticizer, CoarseAggregate, FineAggregate, Age),
                  function(x) c(CompressiveStrength = mean(x$CompressiveStrength)))
head(averaged)
set.seed(975)

forTraining <- createDataPartition(averaged$CompressiveStrength, p=0.75)[[1]]
trainingSet <- averaged[forTraining,]
testSet     <- averaged[-forTraining,]

modFormula <- paste("CompressiveStrength ~ (.)^2 + I(Cement^2) + I(BlastFurnaceSlag^2) +",
                 "I(FlyAsh^2) + I(Water^2) + I(Superplasticizer^2) +",
                 "I(CoarseAggregate^2) + I(FineAggregate^2) + I(Age^2)")
modFormula <- as.formula(modForm)

controlObject <- trainControl(method = 'repeatedcv',
                              repeats = 5,
                              number = 10)
# Linear Models:
linearReg <- train(modFormula,
                   data = trainingSet,
                   method = 'lm',
                   trControl = controlObject)
linearReg

plsModel <- train(modFormula, data = trainingSet,
                  method='pls',
                  preProc = c('center', 'scale'),
                  tuneLength = 15,
                  trControl = controlObject)

enetGrid <- expand.grid(.lambda = c(0, 0.001, 0.01, 0.1),
                        .fraction = seq(0.05, 1, length=20))
enetModel <- train(modForm, data=trainingSet,
                   method = 'enet',
                   preProc = c('center', 'scale'),
                   tuneGrid = enetGrid,
                   trControl = controlObject)

# MARS
earthModel <- train(CompressiveStrength ~ . , data = trainingSet,
                    method = 'earth',
                    tuneGrid = expand.grid(.degree = 1,
                                           .nprune = 2:25),
                    trControl = controlObject)

# Support Vector Machines
svmRModel <-  train(CompressiveStrength ~ . , data=trainingSet,
                    method = 'svmRadial',
                    tuneLength = 15,
                    preProc = c('center', 'scale'),
                    trControl = controlObject)

# Neural Networks
# in repo:
nnetGrid <- expand.grid(decay = c(0.001, .01, .1),
                        size = seq(1, 27, by = 2),
                        bag = FALSE)
nnetFit <- train(CompressiveStrength ~ .,
                 data = trainingSet,
                 method = "avNNet",
                 tuneGrid = nnetGrid,
                 preProc = c("center", "scale"),
                 linout = TRUE,
                 trace = FALSE,
                 maxit = 1000,
                 allowParallel = FALSE,
                 trControl = controlObject)

# in book: 
nnetGrid <- expand.grid(.decay = c(0.001, 0.01, 0.1),
                        .size = seq(1, 27, by = 2),
                        .bag = FALSE)
nnetModel <- train(CompressiveStrength ~ ., data = trainingSet,
                  method = 'avNNet',
                  tuneGrid = nnetGrid,
                  preProc = c('center', 'scale'),
                  trConrol = controlObject,
                  linout = TRUE,
                  trace = FALSE,
                  maxit = 1000)
                  

# TREES
rpartFit <- train(CompressiveStrength ~ . , data = trainingSet,
                  method = "rpart",
                  tuneLength = 30,
                  trControl = controlObject)

treebagFit <- train(CompressiveStrength ~ . , data = trainingSet,
                    method = "treebag",
                    trControl = controlObject)

ctreeFit <- train(CompressiveStrength ~ . , data = trainingSet,
                  method = "ctree",
                  tuneLength = 10,
                  trControl = controlObject)

# random forrest
rfModel <- train(CompressiveStrength ~ ., data = trainingSet,
                 method = 'rf',
                 tuneLength = 10,
                 ntrees = 1000,
                 importance = TRUE,
                 trControl = controlObject)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                       .n.trees = seq(100, 1000, by = 50),
                       .shrinkage = c(0.01, 0.1))

gbmModel <- train(CompressiveStrength ~ ., data = trainingSet,
                 method = 'gbm',
                 tuneGrid = gbmGrid,
                 trControl = controlObject,
                 verbose = FALSE)

cubistGrid <- expand.grid(.committees = c(1,5,10,50,75,100),
                          .neighbors = c(0,1,3,5,7,9))

cbModel <- train(CompressiveStrength ~ ., data = trainingSet,
                 method = 'cubist',
                 tuneGrid = cubistGrid,
                 trControl = controlObject)

allResamples <- resamples(list(
  "Linear Reg" = linearReg
  , "PLS" = plsModel
  , "Elastic Net" = enetModel
  , "MARS" = earthModel
  , "SVM" = svmRModel
  #,"Neural Networks" = nnetModel
  , "Nnet" = nnetFit
  , "CART" = rpartFit
  , "Cond Inf Tree" = ctreeFit
  , "Bagged Tree" = treebagFit
  , "Boosted Tree" = gbmModel
  , "Random Forrest" = rfModel
  , "Cubist" = cbModel
  ))
parallelplot(allResamples)
parallelplot(allResamples, metric = "Rsquared")
xyplot(allResamples, what='tTime')
bwplot(allResamples)
densityplot(allResamples)
densityplot(allResamples, metric="RMSE")
densityplot(allResamples, metric="Rsquared")
dotplot(allResamples)
dotplot(allResamples, metric="RMSE")
dotplot(allResamples, metric="Rsquared")

splom(allResamples) # defaults to RMSE
splom(allResamples, metric= "Rsquared")

# Here, we want to find an optimum set of mixture components for strength. Remove
# outcome data, scale/center and create samples to begin a search of best composition

age28Data <- subset(trainingSet, Age == 28) # Typo in book, trainingSet is right
pp1 <- preProcess(age28Data[, -(8:9)],
                  c('center', 'scale'))
scaledTrain <- predict(pp1, age28Data[, 1:7])
set.seed(91) #??
startMixture <- sample(1:nrow(age28Data), 1)
# produces one starter
starters <- scaledTrain[startMixture, 1:7]
pool <- scaledTrain
index <- maxDissim(starters, pool, 14)
startPoints <- c(startMixture, index)
starters <- age28Data[startPoints, 1:7]

# remove water from the components of the mixture (see pg. 241 for why)
startingValues <- starters[, -4]
modelPrediction <- function(x, mod, limit = 2500)
{
  if(x[1] < 0 | x[1] > 1) return(10^38)
  if(x[2] < 0 | x[2] > 1) return(10^38)
  if(x[3] < 0 | x[3] > 1) return(10^38)
  if(x[4] < 0 | x[4] > 1) return(10^38)
  if(x[5] < 0 | x[5] > 1) return(10^38)
  if(x[6] < 0 | x[6] > 1) return(10^38)
  x <- c(x, 1 - sum(x))
  if(x[7] < 0.05) return(10^38)
  tmp <- as.data.frame(t(x))
  names(tmp) <- c('Cement','BlastFurnaceSlag','FlyAsh',
                  'Superplasticizer','CoarseAggregate',
                  'FineAggregate', 'Water')
  tmp$Age <- 28
  -predict(mod, tmp)
}

# start with Cubist
cbResults <- startingValues
cbResults$Water <- NA
cbResults$Prediction <- NA

for(i in 1:nrow(cbResults)) {
  results <- optim(unlist(cbResults[i,1:6]),
                   modelPrediction,
                   method = "Nelder-Mead",
                   control=list(maxit=5000),
                   mod = cbModel)
  cbResults$Prediction[i] <- -results$value
  cbResults[i,1:6] <- results$par
}

cbResults$Water <- 1 - apply(cbResults[, 1:6], 1, sum)
cbResults <- cbResults[order(-cbResults$Prediction), ][1:3, ]
cbResults$Model <- 'Cubist'
str(cbResults)

nnetResults <- startingValues
nnetResults$Water <- NA
nnetResults$Prediction <- NA
