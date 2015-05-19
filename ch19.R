# CH 19 - Intro to Feature Selection

library('AppliedPredictiveModeling')
data(AlzheimerDisease)

library(doMC)
registerDoMC(7)

str(predictors)

predictors$E2 <- predictors$E4 <- predictors$E5 <- 0
predictors$E2[grepl("2", predictors$Genotype)] <- 1
predictors$E2[grepl("3", predictors$Genotype)] <- 1
predictors$E2[grepl("4", predictors$Genotype)] <- 1

set.seed(730)
split <- createDataPartition(diagnosis, p = 0.8, list = FALSE)
adData <- predictors
adData$Class <- diagnosis
training <- adData[split, ]
testing <- adData[-split, ]

predVars <- names(adData)[!(names(adData) %in% c("Class", "Genotype"))]

fiveStats <- function(...) {
  c(twoClassSummary(...),
    defaultSummary(...))
}

index <- createMultiFolds(training$Class, times = 5)
varSeq <- seq(1, length(predVars) -1, by = 2)

head(training[, predVars])
training$Class

# Forward, Backward and Stepwise

initial <- glm(Class ~ tau + VEGF + E4 + IL_3,
               data = training,
               family = 'binomial')
library("MASS")
# AIC is a penalized version of RMSE
stepAIC(initial, direction = "both")

# Recursive feature Elimination

library(caret)
str(rfFuncs)

newRF <- rfFuncs
newRF$summary <- fiveStats

ctrl <- rfeControl(method = 'repeatdcv',
                   repeats = 5,
                   verbose = TRUE,
                   functions = newRF,
                   index = index)

set.seed(721)
rfRFE <- rfe(x = training[, predVars],
             y = training$Class,
             sizes = varSeq,
             metric = "ROC",
             rfeControl = ctrl,
             ntree = 1000)
rfRFE

# Filter Methods

#not recommended by itself:
pScore <- function(x, y) {
  numX <- length(unique(x))
  if(numX %in% c(0, 1)) {
    out <- 1
  }
  if(numX > 2) {
    out <- t.test(x ~ y)$p.value
  }
  if(numX == 2) {
    out <- fisher.test(factor(x), y)$p.value
  }
  out
}
scores <- apply(X = training[, predVars],
                MARGIN = 2,
                FUN = pScore,
                y = training$Class)
head(scores)

# correct with bonferroni

pCorrection <- function(score, x, y) {
  score <- p.adjust(score, "bonferroni")
  keepers <- (score <= 0.05)
  keepers
}
tail(pCorrection(scores))

# compute with LDA model
ldaWithPvalues <- ldaSBF
ldaWithPvalues$score <- pScore
ldaWithPvalues$summary <- fiveStats
ldaWithPvalues$filter <- pCorrection

sbfCtrl <- sbfControl( method = 'repeatedcv',
                       repeats = 5,
                       verbose = TRUE,
                       functions = ldaWithPvalues,
                       index = index)

ldaFilter <- sbf(training[, predVars],
                 training$Class,
                 tol = 1.0e-12,
                 sbfControl = sbfCtrl)
ldaFilter
