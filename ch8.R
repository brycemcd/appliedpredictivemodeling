# Ch 8, Tree / Rule based models

library(partykit)
library('party')
library(doMC)
registerDoMC(5)

set.seed(100)

# Single Trees:

cartTune <- train(solTrainXtrans, solTrainY,
                   method='rpart2',
                   tuneLength=20,
                   trControl=trainControl(method='cv'))
cartTune
cartTune$finalModel

plot(cartTune, scales = list(x = list(log = 10)))
plot(cartTune$finalModel)
cartTree <- as.party(cartTune$finalModel)
plot(cartTree) # much better plot

cartImp <- varImp(cartTune, scale = FALSE, competes = FALSE)
cartImp
plot(cartImp, top=20)

# Model Trees

# TODO install java :()
m5Tune <- train(solTrainXtrans, solTrainY,
                method = 'M5',
                trControl = trainControl(method='cv'),
                control = Weka_control(M=10))

# Bagged Trees
bagCtrl <- cforest_control(mtry = ncol(solTrainXtrans) - 1)
baggedTree <- cforest(y ~ ., data=solTrainXtrans, controls=bagCtrl)
baggedTree

# in caret:
treebagTune <- train(x = solTrainXtrans, y = solTrainY,
                     method = "treebag",
                     nbagg = 50,
                     trControl = ctrl)
treebagTune
plot(treebagTune)
# random forrest

mtryGrid <- data.frame(mtry = floor(seq(10, ncol(solTrainXtrans), length = 10)))

rfTune <- train(x = solTrainXtrans, y = solTrainY,
                method = "rf",
                tuneGrid = mtryGrid,
                ntree = 1000,
                importance = TRUE,
                trControl = ctrl)
rfTune
plot(rfTune)

rfImp <- varImp(rfTune, scale = FALSE)
rfImp
plot(rfImp, top=20)

# Boosted Trees
gbmGrid <- expand.grid(.interaction.depth = seq(1,7, by=2),
                       .n.trees = seq(100, 1000, by=50),
                       .shrinkage=c(0.01, 0.1))

gbmTune <- train(solTrainXtrans, solTrainY,
                 method='gbm',
                 tuneGrid=gbmGrid,
                 verbose=FALSE)

# Cubist Models
cbGrid <- expand.grid(committees = c(1:10, 20, 50),
                      neighbors = c(5, 9, 13, 25))

cubistTune <- train(solTrainXtrans, solTrainY,
                    method = 'cubist',
                    tuneGrid = cbGrid,
                    trControl = ctrl)

cubistTune
summary(cubistTune)
plot(cubistTune)
plot(cubistTune, auto.key = list(columns = 4, lines = TRUE))
cbImp <- varImp(cubistTune, scale = FALSE)
cbImp
plot(cbImp, top=20)
