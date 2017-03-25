# In this code - 
# Perform logistic regression, classification tree, general additive models and discriminant analysis on given data.
# Calculate various performnce metrics and compare between in-sample and outsample performnce
# Check for over-fitting, and select best performing model

rm(list=ls())
library(ROCR)
library(verification)
library(rpart)
library(mgcv)
library(MASS)
set.seed(10896584)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Data input and split
german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
german_credit$response = german_credit$response - 1

#Create 75% sample
subset <- sample(nrow(german_credit), nrow(german_credit) * 0.75)
train = german_credit[subset, ]
test = german_credit[-subset, ]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Logistic regression
credit.glm0 <- glm(response ~ ., family = binomial(link="logit"), train)
summary(credit.glm0)
BIC(credit.glm0)
credit.glm.step <- step(credit.glm0, direction = c("both")) #Model selection with BIC
summary(credit.glm.step)
BIC(credit.glm.step)

#In sample prediction
prob.glmstep.insample <- predict(credit.glm.step, type = "response")
predicted.glmstep.insample <- prob.glmstep.insample > (1/6)
predicted.glmstep.insample <- as.numeric(predicted.glmstep.insample)

#Miscalculation table and Rate
table(train$response, predicted.glmstep.insample, dnn = c("Truth", "Predicted"))
w1 = 1
w2 = 5
asym_misrate_m1 =  mean(w1*(ifelse((train$response==0 & predicted.glmstep.insample == 1), 1,0)) +
                          w2*(ifelse((train$response == 1 & predicted.glmstep.insample == 0), 1,0)))

#ROC curves and AUC values
roc.plot(x = train$response == "1", pred = cbind(prob.glmstep.insample),
         legend = TRUE, show.thres = FALSE, leg.text = c("GLM"))$roc.vol

#Out sample prediction
prob.glmstep.outsample <- predict(credit.glm.step, test, type = "response")
predicted.glmstep.outsample <- prob.glmstep.outsample > (1/6)
predicted.glmstep.outsample <- as.numeric(predicted.glmstep.outsample)

#Miscalculation table and Rate
table(test$response, predicted.glmstep.outsample, dnn = c("Truth", "Predicted"))
w1 = 1
w2 = 5
asym_misrate_m1_out =  mean(w1*(ifelse((test$response==0 & predicted.glmstep.outsample == 1), 1,0)) +
                          w2*(ifelse((test$response == 1 & predicted.glmstep.outsample == 0), 1,0)))

#ROC curves and AUC values
roc.plot(x = test$response == "1", pred = cbind(prob.glmstep.outsample),
         legend = TRUE, show.thres = FALSE, leg.text = c("GLM"))$roc.vol

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Classification tree
credit.rpart <- rpart(formula = response ~ ., data = train, method = "class", 
                      parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)))
plot(credit.rpart)
text(credit.rpart)

#In sample Prediction
predicted.rpart.insample = predict(credit.rpart, train, type = "class")

#Miscalculation table and Rate
table(train$response, predicted.rpart.insample, dnn = c("Truth", "Predicted"))
asym_misrate_m2 =  mean(w1*(ifelse((train$response==0 & predicted.rpart.insample == 1), 1,0)) +
                          w2*(ifelse((train$response == 1 & predicted.rpart.insample == 0), 1,0)))

#ROC curves and AUC values
roc.plot(x = train$response == "1", pred = cbind(predicted.rpart.insample),
         legend = TRUE, show.thres = FALSE, leg.text = c("Tree"))$roc.vol

#Out sample Prediction
predicted.rpart.outsample = predict(credit.rpart, test, type = "class")

#Miscalculation table and Rate
table(test$response, predicted.rpart.outsample, dnn = c("Truth", "Predicted"))
asym_misrate_m2_out =  mean(w1*(ifelse((test$response==0 & predicted.rpart.outsample == 1), 1,0)) +
                          w2*(ifelse((test$response == 1 & predicted.rpart.outsample == 0), 1,0)))

#ROC curves and AUC values
roc.plot(x = test$response == "1", pred = cbind(predicted.rpart.outsample),
         legend = TRUE, show.thres = FALSE, leg.text = c("Tree"))$roc.vol

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#General additive model
credit.gam <- gam(response ~ chk_acct + s(duration) + credit_his + purpose + s(amount) + 
                    saving_acct + installment_rate + sex + other_debtor + present_resid + 
                    s(age) + other_install + housing + n_people + foreign, family = binomial, data = train)
summary(credit.gam)
credit.gam$deviance/credit.gam$df.residual
plot(credit.gam, shade = TRUE, seWithMean = TRUE, scale = 0)

#In sample Prediction
predicted.gam.insample <- predict(credit.gam, train, type = "response")
predicted.gam.insample <- (predicted.gam.insample >= (1/6)) * 1

#Miscalculation table and Rate
table(train$response, predicted.gam.insample, dnn = c("Truth", "Predicted"))
asym_misrate_m3 =  mean(w1*(ifelse((train$response==0 & predicted.gam.insample == 1), 1,0)) +
                          w2*(ifelse((train$response == 1 & predicted.gam.insample == 0), 1,0)))

#ROC curves and AUC values
roc.plot(x = train$response == "1", pred = cbind(predicted.gam.insample),
         legend = TRUE, show.thres = FALSE, leg.text = c("GAM"))$roc.vol

#Out sample Prediction
predicted.gam.outsample <- predict(credit.gam, test, type = "response")
predicted.gam.outsample <- (predicted.gam.outsample >= (1/6)) * 1

#Miscalculation table and Rate
table(test$response, predicted.gam.outsample, dnn = c("Truth", "Predicted"))
asym_misrate_m3_out =  mean(w1*(ifelse((test$response==0 & predicted.gam.outsample == 1), 1,0)) +
                          w2*(ifelse((test$response == 1 & predicted.gam.outsample == 0), 1,0)))

#ROC curves and AUC values
roc.plot(x = test$response == "1", pred = cbind(predicted.gam.outsample),
         legend = TRUE, show.thres = FALSE, leg.text = c("GAM"))$roc.vol
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Discriminant Analysis
train$response = as.factor(train$response)
credit.lda <- lda(response ~ ., data = train)
summary(credit.lda)

#In sample prediction
prob.lda.insample <- predict(credit.lda, type = "response")
predicted.lda.insample <- (prob.lda.insample$posterior[, 2] >= (1/6)) * 1

#Miscalculation table and Rate
table(train$response, predicted.lda.insample, dnn = c("Truth", "Predicted"))
asym_misrate_m4 =  mean(w1*(ifelse((train$response==0 & predicted.lda.insample == 1), 1,0)) +
                          w2*(ifelse((train$response == 1 & predicted.lda.insample == 0), 1,0)))
#ROC curves and AUC values
roc.plot(x = train$response == "1", pred = cbind(predicted.lda.insample),
         legend = TRUE, show.thres = FALSE, leg.text = c("LDA"))$roc.vol

#Out sample prediction
prob.lda.outsample <- predict(credit.lda, test, type = "response")
predicted.lda.outsample <- (prob.lda.outsample$posterior[, 2] >= (1/6)) * 1

#Miscalculation table and Rate
table(test$response, predicted.lda.outsample, dnn = c("Truth", "Predicted"))
asym_misrate_m4_out =  mean(w1*(ifelse((test$response==0 & predicted.lda.outsample == 1), 1,0)) +
                          w2*(ifelse((test$response == 1 & predicted.lda.outsample == 0), 1,0)))
#ROC curves and AUC values
roc.plot(x = test$response == "1", pred = cbind(predicted.lda.outsample),
         legend = TRUE, show.thres = FALSE, leg.text = c("LDA"))$roc.vol


#ROC curves and AUC values
roc.plot(x = test$response == "1", pred = cbind(prob.glmstep.outsample, predicted.rpart.outsample, predicted.gam.outsample, predicted.lda.outsample),
         legend = TRUE, show.thres = FALSE, leg.text = c("GLM", "Tree", "GAM","LDA"))$roc.vol
