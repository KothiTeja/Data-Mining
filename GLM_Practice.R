# In this code - 
# Perform logit, probit and c-log-log regression using glm function. Compare performance and choose final model
# Improve the model through step-wise varable selection. Check in-sample and outsample performance
# Grid search and cross validation for selcting optimal cut-off

rm(list=ls())
library(verification)
library(ggplot2)
library(corrplot)
library(rpart)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Input data
bankruptcy <- read.csv("bankruptcy.csv", header = T)

#Create 70% sample
subset <- sample(nrow(bankruptcy), nrow(bankruptcy) * 0.7)
train = bankruptcy[subset, ]
test = bankruptcy[-subset, ]

#Run GLM using different link functions
bank.logit <- glm(DLRSN~.-CUSIP-FYEAR, family = binomial(link="logit"), train)
bank.probit <- glm(DLRSN~.-CUSIP-FYEAR, family = binomial(link="probit"), train)
bank.loglog <- glm(DLRSN~.-CUSIP-FYEAR, family = binomial(link="cloglog"), train)

#Compare performance of the models using AIC and BIC values
#Lower AIC & BIC => better model. These are pureley comparative numbers and do not have any meaning of their own
AIC(bank.logit, bank.probit, bank.loglog)
BIC(bank.logit, bank.probit, bank.loglog)
#Logit has least AIC and BIC -> chosen for further work

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Run stepwise model selection on logit; report performance
#Running stepwise selection using AIC and BIC
logit.step <- step(bank.logit) #Step-wise model selection using AIC as factor for improvement
logit.step2 <- step(bank.logit, k = log(nrow(train))) #Step-wise model selection using BIC

AIC(logit.step, logit.step2)
BIC(logit.step, logit.step2)
#Choose model 1 based on AIC

#Residual Deviance
logit.step

#In sample prediction
prob.glmstep.insample <- predict(logit.step, type = "response")
#prob.glmstep2.insample <- predict(logit.step2, type = "response")

#ROC curves and AUC values
roc.plot(train$DLRSN == "1", prob.glmstep.insample)$roc.vol

#Miscalculation table and rate
predicted.glmstep.insample <- prob.glmstep.insample > 0.16
predicted.glmstep.insample <- as.numeric(predicted.glmstep.insample)
table(train$DLRSN, predicted.glmstep.insample, dnn = c("Truth", "Predicted"))
mean(ifelse(train$DLRSN != predicted.glmstep.insample, 1, 0))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Out-of-sample performance
#Outof sample prediction
prob.glmstep.outsample <- predict(logit.step,test, type = "response")

#Miscalculation table and rate
predicted.glmstep.outsample <- prob.glmstep.outsample > 0.09
predicted.glmstep.outsample <- as.numeric(predicted.glmstep.outsample)
table(test$DLRSN, predicted.glmstep.outsample, dnn = c("Truth", "Predicted"))
#Asym miscalc rate
w1 = 1
w2 = 15
mean(w1*(ifelse((test$DLRSN==0 & predicted.glmstep.outsample == 1), 1,0)) +
                          w2*(ifelse((test$DLRSN == 1 & predicted.glmstep.outsample == 0), 1,0)))

#ROC curves and AUC values
roc.plot(test$DLRSN == "1", prob.glmstep.outsample)$roc.vol

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Grid search for cut off rates
#Cost function - in the cost function, both r and pi are vectors, r=truth, pi=predicted probability
cost1 <- function(r, pi) {
  weight1 = 15
  weight0 = 1
  c1 = (r == 1) & (pi < pcut)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi > pcut)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

########Grid search for optimal cut-off
# define the search grid from 0.01 to 0.99
searchgrid = seq(0.01, 0.99, 0.01)
# result is a 99x2 matrix, the 1st col stores the cut-off p, the 2nd column stores the cost
result = cbind(searchgrid, NA)
for (i in 1:length(searchgrid)) {
  pcut <- result[i, 1]
  # assign the cost to the 2nd col
  result[i, 2] <- cost1(train$DLRSN, prob.glmstep.insample)
}
plot(result, ylab = "Cost in Training Set")
grid.prob <- result[which(result[,2] == min(result[,2]), arr.ind=TRUE),]
#This gives optimal probability as 8%. Will change with costs.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Cross-validation for optimal cutoff using entire data set
########Cross validate for optimal cut-off
searchgrid2 = seq(0.01, 0.6, 0.02)
result2 = cbind(searchgrid2, NA)
cv.logit <- step(glm(DLRSN~.-CUSIP-FYEAR, family = binomial(link="logit"), bankruptcy))
for (i in 1:length(searchgrid2)) {
  pcut <- result2[i, 1]
  result2[i, 2] <- cv.glm(data = bankruptcy, glmfit = cv.logit, cost = cost1, 
                         K = 3)$delta[2]
}
plot(result2, ylab = "CV Cost")
cv.prob <- result2[which(result2[,2] == min(result2[,2]), arr.ind=TRUE),]#Asymm miscalc rate

#ROC curves and AUC values
prob.cv<-predict(cv.logit,bankruptcy, type = "response")
roc.plot(bankruptcy$DLRSN == "1", prob.cv)$roc.vol

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Classification tree
#Classification tree
bank.rpart <- rpart(formula = DLRSN~.-CUSIP-FYEAR, data = train, method = "class", 
                      parms = list(loss = matrix(c(0, 15, 1, 0), nrow = 2)))
#Asymm miscalc rate
pred.tree = predict(bank.rpart, test, type = "class")
table(test$DLRSN, pred.tree, dnn = c("Truth", "Predicted"))
mean(w1*(ifelse((test$DLRSN==0 & pred.tree == 1), 1,0)) +
       w2*(ifelse((test$DLRSN == 1 & pred.tree == 0), 1,0)))

#ROC curves and AUC values
prob.rpart<-predict(bank.rpart,train, type = "prob")
roc.plot(train$DLRSN == "1", prob.rpart[,2])$roc.vol
