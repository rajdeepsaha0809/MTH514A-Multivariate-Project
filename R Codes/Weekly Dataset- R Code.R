rm(list= ls())
set.seed(201380)
library(ISLR)
data <- Weekly
str(data)
head(data)
attach(data)
#Checking for missing values
sum(is.na(data))
#Check for Data Imbalance
attach(data)
sum(Direction == "Up") / nrow(data)
sum(Direction == "Down") / nrow(data)
#Data Split
names(data)
index= sample(1:nrow(data),floor(0.85*nrow(data)))
train= data[index, ]
remaining= data[-index, ]
index2 = sample(1:nrow(remaining),floor(2/3*nrow(remaining)))
crossval= remaining[index2, ]
test = remaining[-index2, ]
actual_Direction=crossval$Direction
dim(train)
dim(crossval)
dim(test)
library(tibble)
library(cvms)
f_cfm <- function(x){
  cfm <- as.tibble(x)
  cname <- colnames(cfm)
  print(plot_confusion_matrix(cfm, target_col = cname[2], prediction_col =  cname[1], counts_col = cname[3]))
}
#Logistic Regression
attach(data)
Direction <- as.factor(Direction)
logistic.fit=glm(Direction~Year+Lag1+Lag2+Lag3+Lag4+Lag5+Volume,family=binomial,data=train)
logistic.probs <- predict(logistic.fit, crossval, type = "response")
logistic.pred <- rep("Down", nrow(crossval))
logistic.pred[logistic.probs > 0.5] = "Up"
actual <- crossval$Direction
logistic_table <- table(logistic.pred, actual)
f_cfm(logistic_table)
#Normality Test
library(Hmisc)
hist.data.frame(train[,1:8])
library(MASS)
lda.fit= lda(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Today,data= train)
lda.pred= predict(lda.fit, crossval, type="response")$class
lda_table <- table(lda.pred, actual)
f_cfm(lda_table)
qda.fit= qda(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Today,data= train)
qda.pred= predict(qda.fit, crossval, type="response")$class
qda_table <- table(qda.pred, actual)
f_cfm(qda_table)
#Decision Tree
library(tree)
tree.fit <- tree(Direction~., data= train)
cv.Direction <- cv.tree(tree.fit, FUN= prune.misclass)
cv.Direction #dev corresponds to misclassification error rate
par(mfrow=c(1,2))
plot(cv.Direction$size, cv.Direction$dev, type= "b")
plot(cv.Direction$k, cv.Direction$dev, type= "b")
prune.tree <- prune.misclass(tree.fit, best = 2 )
plot(prune.tree)
text(prune.tree, pretty = 0)
tree.pred <- predict(prune.tree, crossval, type = "class")
tree_table <- table(tree.pred, actual)
f_cfm(tree_table)
#Random Forest
library(randomForest)
used_pred <- floor(sqrt(ncol(data)-1))
rf.fit <- randomForest(Direction~., data = train, mtry = used_pred,
           importance = TRUE, maxdepth = 8)
rf.pred <- predict(rf.fit, newdata = crossval)
rf_table <- table(rf.pred, actual)
f_cfm(rf_table)
logistic_table
lda_table
qda_table
tree_table
rf_table
F_score <- function(x){
  p <- x[2, 2]/(x[2, 2]+ x[2, 1])
  r <- x[2, 2]/(x[2, 2]+ x[1, 2])
  f <- round(2*p*r/(p+r), 4)
  f
}
F_score(logistic_table)
F_score(lda_table)
F_score(qda_table)
F_score(tree_table)
F_score(rf_table)
paste("We have got the maximum F-Score for Random Forest which is", round(F_score(rf_table), 4))
#Plotting ROC curve

#Logistic Regression
library(pROC)
logistic_prob = predict(logistic.fit, newdata = crossval, type = "response")
logistic_roc = roc(crossval$Direction ~ logistic_prob, plot = TRUE, main= "ROC Curve for Logistic Regression", print.auc = TRUE)
paste("Area under the curve is",round(auc(logistic_roc), 5))

#LDA
lda_prob= predict(lda.fit, crossval, type="response")$posterior[,2]
lda_roc = roc(crossval$Direction ~ lda_prob, plot = TRUE, main= "ROC Curve for LDA", print.auc = TRUE)
paste("Area under the curve is",round(auc(lda_roc), 5))

#LDA
qda_prob= predict(qda.fit, crossval, type="response")$posterior[,2]
qda_roc = roc(crossval$Direction ~ qda_prob, plot = TRUE, main= "ROC Curve for QDA", print.auc = TRUE)
paste("Area under the curve is",round(auc(qda_roc), 5))

#Decision Tree
tree_predict= predict(prune.tree, crossval, type="vector")
tree_roc = roc(crossval$Direction~tree_predict[,2], plot = TRUE, main= "ROC Curve for Decision Tree", print.auc = TRUE)
paste("Area under the curve is",round(auc(tree_roc), 10))

#Random Forest
rf_predict= predict(rf.fit, crossval, type="prob")
rf_roc = roc(crossval$Direction~rf_predict[,2], plot = TRUE, main= "ROC Curve for Random Forest", print.auc = TRUE)
paste("Area under the curve is",round(auc(rf_roc), 10))
#Final Fit
final_fit <- predict(rf.fit, newdata = test)
actual_test <- test$Direction
final_rf_table <- table(final_fit, actual_test)
f_cfm(final_rf_table)
#Evaluation Metric for Test Set

#F_Score
F_score(final_rf_table)

rf_predict_test= predict(rf.fit, test, type="prob")
rf_roc_test = roc(test$Direction~rf_predict_test[,2], plot = TRUE, main= "ROC Curve for Final Model",  print.auc = TRUE)
paste("Area under the curve is",round(auc(rf_roc_test),4))