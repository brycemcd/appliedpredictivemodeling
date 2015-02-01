#CH7 - nonlinear regression models

library(caret)
library(earth)
library(kernlab)
library(nnet)

library(doMC)
registerDoMC(4)
# Neural Nets:

# remove predictors where pairwise correlation is high
set.seed(100)
tooHigh <- findCorrelation(cor(solTrainXtrans), cutoff=.75)
trainXnnet <- solTrainXtrans[, -tooHigh]
testXnnet <- solTestXtrans[, -tooHigh]

nnetGrid <- expand.grid(.decay  = c(0, 0.01, 0.1),
                        .size   = c(1:10),
                        .bag    = FALSE)
ctrl <- trainControl(method='cv', number=10)

nnetTune <- train(solTrainXtrans, solTrainY,
                  method = 'avNNet',
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  preProc = c('center', 'scale'),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(trainXnnet) + 1) + 10 + 1,
                  maxit = 500
                  )

# MARS !

#simple model:
marsFit <- earth(solTrainXtrans, solTrainY)
summary(marsFit)
plot(marsFit)

# Now do it with the caret package
marsGrid <- expand.grid(.degree = 1:2,
                        .nprune = 2:38)
marsTuned <- train(solTrainXtrans, solTrainY,
                   method='earth',
                   tuneGrid=marsGrid,
                   trControl=trainControl(method='cv'))
marsTuned
# show variable importance:
varimp <- varImp(marsTuned)
# show all vars where importance is > 0
numvars <- nrow(subset(varimp$importance, Overall > 0))
plot(varimp, top=numvars)


# Support Vector Machines
svmLin <- train(solTrainXtrans, solTrainY,
                method='svmLinear',
                preProc=c('center', 'scale'),
                tuneLength=14,
                trControl = ctrl
                )


svmRad <- train(x = solTrainXtrans, y = solTrainY,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneLength=14,
                  trControl = ctrl)
plot(svmRad,
     scales = list(x = list(log = 2),
                   between = list(x = .5, y = 1)))

polyGrid <- expand.grid(degree = 1:2,
                       scale = c(0.01, 0.005, 0.001),
                       C = 2^(-2:5))
svmPoly <- train(x = solTrainXtrans, y = solTrainY,
                  method = "svmPoly",
                  preProc = c("center", "scale"),
                  tuneGrid = polyGrid,
                  trControl = ctrl)
plot(svmPoly,
     scales = list(x = list(log = 2),
                   between = list(x = .5, y = 1))) 

# K-Nearest Neighbors

knnDescr <- solTrainXtrans[, -nearZeroVar(solTrainXtrans)]
knnTune <-train(knnDescr, solTrainY,
                method='knn',
                preProc=c('center', 'scale'),
                tuneGrid=data.frame(.k=1:20),
                trControl=ctrl)
knnTune
plot(knnTune)