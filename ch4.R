# Chapter 4
install.packages(c('AppliedPredictiveModeling', 'caret', 'Design', 'e1071', 'ipred', 'MASS'))

library('AppliedPredictiveModeling')
library('caret')
data(twoClassData)

str(predictors)
str(classes)
set.seed(1)
trainingRows <- createDataPartition(classes,
                                   p=0.80,
                                   list=FALSE)
head(trainingRows)
trainPredictors <- predictors[trainingRows,]
trainClasses    <- classes[trainingRows]

testPredictors <- predictors[-trainingRows,]
testClasses    <- classes[-trainingRows]

str(trainPredictors)

# create multiple training/test sets
repeatSplits <- createDataPartition(trainClasses,
                                    p=0.80,
                                    times=3)
str(repeatSplits)

# createResamples = bootstrap
# createFolds = k-fold cross validation
# createMultiFolds = repeated cross validation

cvSplits <- createFolds(trainClasses, k=10, returnTrain = TRUE)
fold1 <- cvSplits[[1]]
cvPredictors <- trainPredictors[fold1,]
cvClasses1 <- trainClasses[fold1]

# Tuning a Model
data(GermanCredit)
# From https://github.com/cran/AppliedPredictiveModeling/blob/master/inst/chapters/04_Over_Fitting.R
set.seed(100)
inTrain <- createDataPartition(GermanCredit$Class, p = .8)[[1]]
GermanCreditTrain <- GermanCredit[ inTrain, ]
GermanCreditTest  <- GermanCredit[-inTrain, ]

set.seed(1056)
# basic
svmFit <- train(Class ~ ., data=GermanCreditTrain, method="svmRadial")
svmFit
# with preprocessing
svmFit <- train(Class ~ .,
                data=GermanCreditTrain, 
                method="svmRadial",
                preProc = c('center', 'scale'))
svmFit
# with eval'ing cost values from 2^-2 to 2^7
svmFit <- train(Class ~ .,
                data=GermanCreditTrain, 
                method="svmRadial",
                preProc = c('center', 'scale'),
                tuneLength = 10)
svmFit
# adjust performance with repeated cross validation instead of the default
# bootstrap

svmFit <- train(Class ~ .,
                data=GermanCreditTrain, 
                method="svmRadial",
                preProc = c('center', 'scale'),
                tuneLength = 10,
                trControl = trainControl(method= 'repeatedcv', repeats=5, classProbs = TRUE))
svmFit
# we can plot the performance!
plot(svmFit, scales= list(x=list(log = 2)))

# now, predict on the test set:
# with qualitative yes/no
predictedClasses <- predict(svmFit, GermanCreditTest)
str(predictedClasses)
# with probabilities:
predictedProbs <- predict(svmFit, newdata=GermanCreditTest, type="prob")
head(predictedProbs)

# now try a logistic regression:
logisticReg <- train(Class ~ .,
                data=GermanCreditTrain, 
                method="glm",
                trControl = trainControl(method= 'repeatedcv', repeats=5, classProbs = TRUE))

logisticReg

# compare models:
resamp <- resamples(list(SVM=svmFit, 
                         Logistic=logisticReg))
summary(resamp)
summary(diff(resamp))
