# used in CH 12 + Ch 13
ctrl <- trainControl( method = 'LGOCV',
                      summaryFunction = twoClassSummary,
                      classProbs = TRUE,
                      index = list(TrainSet = pre2008),
                      savePredictions = TRUE)