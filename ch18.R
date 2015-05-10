# Ch 18 - measuring predictor importance
library('AppliedPredictiveModeling')
library('minerva')
library("CORElearn")

data(solubility)
cor(solTrainXtrans$NumCarbon, solTrainY)

# columns with "FP" in them are categorical
fpCols <- grepl("FP", names(solTrainXtrans))
numericPreds <- names(solTrainXtrans)[!fpCols]
catPreds <- names(solTrainXtrans)[fpCols]

corrValues <- apply(solTrainXtrans[, numericPreds],
                    MARGIN = 2,
                    FUN = function(x, y) cor(x, y),
                    y = solTrainY)
head(corrValues)
plot(corrValues)

corrValuesSpearman <- apply(solTrainXtrans[, numericPreds],
                      MARGIN = 2,
                      FUN = function(x, y) cor(x, y, method = 'spearman'),
                      y = solTrainY)
head(corrValuesSpearman)
plot(corrValuesSpearman)

smoother <- loess(solTrainY ~ solTrainXtrans$NumCarbon)
smoother
xyplot(solTrainY ~ solTrainXtrans$NumCarbon,
       type = c('p', 'smooth'),
       xlab ='# Carbons',
       ylab = 'Solubility'
       )
loessResults <- filterVarImp(x = solTrainXtrans[, numericPreds],
                             y = solTrainY,
                             nonpara = TRUE)
head(loessResults)

# MIC
micValues <- mine(solTrainXtrans[, numericPreds], solTrainY)

names(micValues)
head(micValues$MIC)
head(micValues$MICR2)

# categorical outcomes can be t-tested easily:
t.test(solTrainY ~ solTrainXtrans$FP044)

getTstats <- function(x, y) {
  tTest <- t.test(y~x)
  out <- c(tStat = tTest$statistic, p = tTest$p.value)
  out
}
tVals <- apply(solTrainXtrans[, fpCols],
               MARGIN = 2,
               FUN = getTstats,
               y = solTrainY)
tVals
dim(tVals)
# switch dimensions
tVals <- t(tVals)
head(tVals)

## TODO volccn plpoot?

# Categorical Outcomes

data(segmentationData)
cellData <- subset(segmentationData, Case == "Train")
cellData$Case <- cellData$Cell <- NULL
head(names(cellData))

rocValues <- filterVarImp(x = cellData[, -1],
                          y = cellData$Class)
head(rocValues)

reliefValues <- attrEval(Class ~ .,
                         data = cellData,
                         estimator = "ReliefFequalK",
                         ReliefIterations = 50)
?attrEval # for estimator functions above
head(reliefValues)

perm <- permuteRelief(x = cellData[, -1],
                      y = cellData[, 1],
                      nperm = 500,
                      estimator = 'ReliefFequalK',
                      ReliefIterations = 50)
head(perm$permutations)
histogram( ~value|Predictor,
           data = perm$permutations)
head(perm$standardized)

micValues <- mine(x = cellData[, -1],
                  y = ifelse(cellData$Class == "PS", 1, 0))
head(micValues$MIC)

# Odds Ratio
View(solTrainXtrans[, catPreds])
head(catPreds)
class(solTrainXtrans$FP004)

# 2x2 tables produce different output with the fisher test
filterTrain <- subset(training, AWAOREG %in% c(0, 1))
spTable <- table(filterTrain$AWAOREG,
                 filterTrain$CARAVAN)
spTable
fisher.test(spTable)

# 2xN tables just produce brief output, but p value can still be used
ciTable <- table(training$ABRAND, # 86 = Caravan
                 training$CARAVAN)
ciTable
fisher.test(ciTable)
chisq.test(ciTable)

# caret has a varImp function:

library('randomForest')
rfImp <- randomForest(Class ~ .,
                      data = cellData,
                      ntree = 200,
                      importance = TRUE)
head(varImp(rfImp))
