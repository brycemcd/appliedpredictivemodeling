# CH 11

library("AppliedPredictiveModeling")
library("caret")
library("klaR")
library("MASS")
library("pROC")
library("randomForest")

# generate simulated data

set.seed(975)
simulatedTrain <- quadBoundaryFunc(500)
simulatedTest  <- quadBoundaryFunc(1000)

# fit random forest and quadradic discriminant model

rfModel <- randomForest(class ~ X1 + X2,
                        data = simulatedTrain,
                        ntree = 2000)

qdaModel <- qda(class ~ X1 + X2,
                data = simulatedTrain)

qdaTrainPred <- predict(qdaModel, simulatedTrain)
str(qdaTrainPred)
head(qdaTrainPred$posterior)

qdaTestPred <- predict(qdaModel, simulatedTest)
simulatedTrain$QDAprob <- qdaTrainPred$posterior[, 'Class1']
simulatedTest$QDAprob  <- qdaTestPred$posterior[, 'Class1']

# return probabilities of each class
rfTestPred <- predict(rfModel, simulatedTest, type='prob')
head(rfTestPred)
simulatedTest$RFprob <- rfTestPred[, 'Class1']

# return predicted class
simulatedTest$RFclass <- predict(rfModel, simulatedTest)
head(simulatedTest$RFclass)

# caret can calculate sensitivity and specificity!
sensitivity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            positive = 'Class1')

specificity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            negative = 'Class2')

posPredValue(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            positive = 'Class1')

negPredValue(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            positive = 'Class2')

# caret can do confusion matrix too!

# most analyses should just start with this function. It provides the PPV, NPV
# and an assortment of other good data to know if the model is performing to
# the goals of the analysis

confusionMatrix(data = simulatedTest$RFclass,
                reference = simulatedTest$class,
                positive = 'Class1')


# generating ROC curves

# create plottable object
rocCurve <- roc(response = simulatedTest$class,
                predictor = simulatedTest$RFprob,
                # reverse labels
                levels = rev(levels(simulatedTest$class)))
head(rocCurve)
str(rocCurve)

auc(rocCurve)
ci.roc(rocCurve)

plot(rocCurve, legacy.axes = TRUE)

# lift charts:

labs <- c(RFprob = 'Random Forest',
          QDAprob = 'Quadradic Disciminant Analysis')

liftCurve <- lift(class ~ RFprob + QDAprob,
                  data = simulatedTest,
                  labels = labs)
liftCurve
xyplot(liftCurve,
       auto.key = list(columns = 2,
                       lines = TRUE,
                       points = FALSE))

# calibration plots

# build the object
calCurve <- calibration(class ~ RFprob + QDAprob, data = simulatedTest)
calCurve

xyplot(calCurve, auto.key = list(columns = 2))

# try fitting a sigmoid plot using the glm package

sigmoidalCal <- glm(relevel(class, ref = "Class2") ~ QDAprob,
                    data = simulatedTrain,
                    family = binomial)
coef(summary(sigmoidalCal))

sigmoidProbs <- predict(sigmoidalCal,
                        newdata = simulatedTest[, "QDAprob", drop = FALSE],
                        type = "response")
simulatedTest$QDAsigmoid <- sigmoidProbs

BayesCal <- NaiveBayes(class ~ QDAprob, data = simulatedTrain,
                       usekernel = TRUE)
BayesProbs <- predict(BayesCal,
                      newdata = simulatedTest[, 'QDAprob', drop = FALSE])
simulatedTest$QDABayes <- BayesProbs$posterior[, 'Class1']
head(BayesProbs$posterior)
head(simulatedTest[, c(5:6, 8, 9)])

calCurve2 <- calibration(class ~ QDAprob + QDABayes + QDAsigmoid,
                         data = simulatedTest)
xyplot(calCurve2)

# interesting ... play with the lift curve a little more
labs <- c(QDABayes = 'Bayes - QDA',
          QDAsigmoid= 'Quadradic Disciminant Analysis w/ sigmoid',
          RFprob = 'Random Forest',
          QDAprob = 'Quadradic Disciminant Analysis')

liftCurve <- lift(class ~ QDABayes + QDAsigmoid + RFprob + QDAprob,
                  data = simulatedTest,
                  labels = labs)
liftCurve
xyplot(liftCurve,
       auto.key = list(columns = 2,
                       lines = TRUE,
                       points = FALSE))


# play around with the ROC curve with the various models created:

rocCurve.rfprob <- roc(response = simulatedTest$class,
                predictor = simulatedTest$RFprob,
                # reverse labels
                levels = rev(levels(simulatedTest$class)))


rocCurve.qdaprob <- roc(response = simulatedTest$class,
                predictor = simulatedTest$QDAprob,
                # reverse labels
                levels = rev(levels(simulatedTest$class)))

rocCurve.qdabayes <- roc(response = simulatedTest$class,
                predictor = simulatedTest$QDABayes,
                # reverse labels
                levels = rev(levels(simulatedTest$class)))

rocCurve.qdasigmoid <- roc(response = simulatedTest$class,
                predictor = simulatedTest$QDAsigmoid,
                # reverse labels
                levels = rev(levels(simulatedTest$class)))

plot(rocCurve.rfprob, legacy.axes = TRUE) # from the book
plot.roc(rocCurve.qdaprob, legacy.axes = TRUE, add = TRUE,
         col = 'red')
plot.roc(rocCurve.qdabayes, legacy.axes = TRUE, add = TRUE,
         col = 'blue')
plot.roc(rocCurve.qdasigmoid, legacy.axes = TRUE, add = TRUE,
         col = 'green')
