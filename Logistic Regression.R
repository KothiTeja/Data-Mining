# In this code - 
# Perform logistic regression using glm function.
# Improve the model through step-wise varable selection using AIC and BIC as deciding factor.
# Compare in-sample and outsample performance
# Calculate various performance metrics - Confusion matrix, ROC curves and AUC, and misclassification rate

rm(list=ls())
library(verification)
library(ggplot2)
library(corrplot)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Download data
german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
german_credit$response = german_credit$response - 1

#Create 70% sample
subset <- sample(nrow(german_credit), nrow(german_credit) * 0.7)
train = german_credit[subset, ]
test = german_credit[-subset, ]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Summary and EDA
summary(train)
dim(german_credit)
train_num = train[,c("duration","age")]
ggplot(stack(train_num), aes(x = ind, y = values)) + geom_boxplot()
amount <- data.frame(train[,"amount"])
ggplot(amount, aes(x = "", y = amount)) + geom_boxplot()

#Correlation
#train_num2 = train[,colnames(train)[sapply(train,class)==c("numeric","integer")]]
train_num2 = train[,c("duration","amount","installment_rate","age","response")]
a<-cor(train_num2)
corrplot(a, method="circle")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Logistic regression
credit.glm0 <- glm(response ~ ., family = binomial(link="logit"), train)
summary(credit.glm0)
BIC(credit.glm0)

credit.glm.step <- step(credit.glm0) #Stepwise variable selection
summary(credit.glm.step)
BIC(credit.glm.step)

credit.glm.step2 <- step(credit.glm0, k = log(nrow(train))) #Model selection with BIC
summary(credit.glm.step2)
BIC(credit.glm.step2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#In sample prediction
prob.glmstep.insample <- predict(credit.glm.step, type = "response")
predicted.glmstep.insample <- prob.glmstep.insample > 0.2
predicted.glmstep.insample <- as.numeric(predicted.glmstep.insample)

prob.glmstep2.insample <- predict(credit.glm.step2, type = "response")
predicted.glmstep2.insample <- prob.glmstep2.insample > 0.2
predicted.glmstep2.insample <- as.numeric(predicted.glmstep2.insample)

#Outof sample prediction
prob.glmstep.outsample <- predict(credit.glm.step,test, type = "response")
predicted.glmstep.outsample <- prob.glmstep.outsample > 0.2
predicted.glmstep.outsample <- as.numeric(predicted.glmstep.outsample)

prob.glmstep2.outsample <- predict(credit.glm.step2,test, type = "response")
predicted.glmstep2.outsample <- prob.glmstep2.outsample > 0.2
predicted.glmstep2.outsample <- as.numeric(predicted.glmstep2.outsample)

#Miscalculation tables
table(train$response, predicted.glmstep.insample, dnn = c("Truth", "Predicted"))
mean(ifelse(train$response != predicted.glmstep.insample, 1, 0))
table(train$response, predicted.glmstep2.insample, dnn = c("Truth", "Predicted"))
mean(ifelse(train$response != predicted.glmstep2.insample, 1, 0))

table(test$response, predicted.glmstep.outsample, dnn = c("Truth", "Predicted"))
table(test$response, predicted.glmstep2.outsample, dnn = c("Truth", "Predicted"))

#ROC curves and AUC values
roc.plot(x = train$response == "1", pred = cbind(prob.glmstep.insample, prob.glmstep2.insample),
         legend = TRUE, show.thres = FALSE, leg.text = c("Model 1", "Model 2"))$roc.vol

roc.plot(x = test$response == "1", pred = cbind(prob.glmstep.outsample),
         legend = TRUE, show.thres = FALSE, leg.text = c("Model 1"))$roc.vol

#Asymmetric miscalculation rate
w1 = 1
w2 = 5
asym_misrate_q3 =  mean(w1*(ifelse((test$response==0 & predicted.glmstep.outsample == 1), 1,0)) +
                          w2*(ifelse((test$response == 1 & predicted.glmstep.outsample == 0), 1,0)))
asym_misrate_q3
