#Try to predict the type of wine (type) based on the following variables: fixed acidity,
#volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur
#dioxide, density, pH, sulphates and alcohol. Use these prediction techniques:
#  - logistic regression
#- lasso logistic regression
#- linear discriminant analysis
#- quadratic discriminant analysis
#- naïve Bayes estimation
#- k nearest neighbor
#- support vector machine

#What is the greatest prediction accuracy (in the test set) that you can get?



WineQ <-read.csv('winequality.csv')
str(WineQ)
WineQ$quality <-NULL

Reg_Model <- glm(type~volatile.acidity+citric.acid+residual.sugar+
                   chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+sulphates+alcohol, data = WineQ, family=binomial())
summary(Reg_Model)

library(car)
vif(Reg_Model)

#Fixed.acidity and pH removed

##Compute the Antilogs of the coefficients 

expb <-exp(coef(Reg_Model))
print(expb)

#####

#Predict probability for Good or Bad credit
pred_probs <- predict(Reg_Model, type="response")
head(pred_probs)

#Preditct Red or white wine
pred <-ifelse(pred_probs<0.5, "Red","White")
head(pred)

table(WineQ$type, pred)

Accuracy <-mean(pred == WineQ$type)
Accuracy

library(ROCR)

#Plot the Roc Curve
pr <-prediction(pred_probs, WineQ$type)
perf<-performance(pr, x.measure = "fpr", measure = "tpr")
plot(perf)

auc <-performance(pr, measure="auc")
auc
##99%

######Validate 

n<-sample(6497, 3200)
wine_train <- WineQ[n, ]
wine_test <-WineQ[-n, ]

fit_train <- glm(type~volatile.acidity+citric.acid+residual.sugar+
                   chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+sulphates+alcohol, data = wine_train, family=binomial())

###Compute the prediction accuracy on the test set 
###Estimate probabilities of Red or white

pred_probs <-predict.glm(fit_train, newdata = wine_test, type="response")
head(pred_probs)

pred <- ifelse(pred_probs<0.5, "Red", "White")
head(pred)

table(wine_test$type, pred)

accuracy <-mean(pred==wine_test$type)

#Build Roc Curve for Test set
pr <-prediction(pred_probs, wine_test$type)
perf<-performance(pr, x.measure = "fpr", measure = "tpr")
plot(perf)

auc <-performance(pr, measure="auc")
auc
#99%

########
## Lasso Logistic Regression 
library(glmnet)

x <-model.matrix(type~fixed.acidity+pH+volatile.acidity+citric.acid+residual.sugar+
                   chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+sulphates+alcohol, WineQ)[,-1]
y<-WineQ$type

##Lambda values are power of 10
w <- seq(10,-3, length= 100)
lvalues <-10^w
lvalues

fit <-glmnet(x,y,alpha = 1, lambda = lvalues, family = "binomial")

fit$lambda[88]

coef(fit)[, 88]
coef(fit)[, 88][coef(fit)[,88]!=0]

#Validate lasso logistic regression 

n<-sample(6497, 3200)

cv_fit <- cv.glmnet(x[n,], y[n], alpha=1, nfolds=10, family="binomial")

optLambda <- cv_fit$lambda.min
optLambda

#Predict probability for test set 
pred.probs <- predict(cv_fit, s=optLambda, newx=x[-n,], type="response")
head(pred.probs)

estclass <-ifelse(pred.probs<0.5, "Red", "white")
head(estclass)

#prediction accuracy 
acc <-mean(estclass==wine_test$type)
acc
###75% accuracy

predict(fit, s=optLambda, type="coefficients")
plot(fit)

######Linear Discriminant Analysis

library(MASS)

fit <-lda(type~fixed.acidity+pH+volatile.acidity+citric.acid+residual.sugar+
            chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+sulphates+alcohol, data=WineQ)
#Standardized Coefficients
fit

pred <- predict(fit) #list of predicted 
head(pred)

class<-pred$class #The estimated class
head(class)

table(WineQ$type, class)

correct <-mean(WineQ$type==class)
correct
##99%

#### Validate Linear Discriminant Analysis

n <-sample(6497, 3200)

fit <-lda(type~fixed.acidity+pH+volatile.acidity+citric.acid+residual.sugar+
            chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+sulphates+alcohol, data=wine_train)
#Predict on the test set 
pred_test <- predict(fit, newdata=wine_test)

#list of predicted values
class <-pred_test$class
head(class)

correct <-mean(wine_test$type==class)
correct
##99%

plot(fit)

####Quadratic Discriminant Analysis

fit <-qda(type~fixed.acidity+pH+volatile.acidity+citric.acid+residual.sugar+
            chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+sulphates+alcohol, data=WineQ)

##Correctly classified cases
pred <- predict(fit)
estclass <-pred$class
head(estclass)

correct <-mean(estclass==WineQ$type)
correct
##98%


###Validate the Quadratic Discriminant Analysis

fit <-qda(type~fixed.acidity+pH+volatile.acidity+citric.acid+residual.sugar+
            chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+sulphates+alcohol, data=wine_train)

##Compute prediction accuracy on the test set 
pred_test <- predict(fit, newdata = wine_test)

class <-pred_test$class

correct <- mean(wine_test$type==class)
correct
#98%

#####Naive Bayes 

library(e1071)

bayes <-naiveBayes(type~fixed.acidity+pH+volatile.acidity+citric.acid+residual.sugar+
                     chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+sulphates+alcohol, data=wine_train)

bayes

pred <-predict(bayes, wine_test)
head(pred)

table(wine_test$type, pred)



correct <- mean(wine_test$type==class)
correct
#98%

###KNN

#Check for missing values 
sapply(WineQ, function(x) sum(is.na(x)))

#standardize the predictors 
WineQ2 <- data.frame(scale(WineQ[2:11]))

n <-sample(6497, 3200)
wine_train <- WineQ2[n, ]
wine_test <-WineQ2[-n, ]
library(class)

pred <- knn(train=wine_train[,-1], test = wine_test[,-1], cl=type, k=10)
str(WineQ)

library(caret)
x <- trainControl(method = "svmRadial",
                classProbs = TRUE, 
                repeats = 5)
            
knn_model <- train(type~., data=WineQ, 
                  method="svmRadial",
                  preProc=c("center", "scale"),
                  tuneLength=5, 
                  trControl= x)
knn_model                

plot(knn_model)
                  
library(e1071)

T_knn <- tune.knn(wine_train[,-1], factor(wine_train[,1]),k=1:100)
T_knn

plot(T_knn)


###Suport Vector Machine

n<-sample(6497, 3200)
wine_train <- WineQ[n, ]
wine_test <-WineQ[-n, ]

fit <- svm(type~., data=wine_train, 
           type= "C-classification", kernel="linear", cost=16)
pred <-predict(fit, wine_test)
mean(pred==wine_test$type)
#99% accuracy

###To find best cost (Improve accuracy)
###10 fold cv 
t_lin <- tune.svm(type~., data=wine_train, cost = 2^(2:8), kernal="linear")
t_lin$best.parameters


