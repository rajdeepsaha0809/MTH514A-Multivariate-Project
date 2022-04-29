#Loading the Data

rm(list=ls())
set.seed(201380)
data<- read.csv("https://raw.githubusercontent.com/rajdeepsaha0809/MTH514A-Multivariate-Project/main/BankNote_Authentication.csv")
head(data)
dim(data)
str(data)
data$class <- as.factor(data$class)

#Checking for missing values
sum(is.na(data))

#Check for Data Imbalance
attach(data)
sum(class == 1) / nrow(data)
sum(class == 0) / nrow(data)

#Data Split
names(data)
index= sample(1:nrow(data),floor(0.85*nrow(data)))
train= data[index, ]
remaining= data[-index, ]
index2 = sample(1:nrow(remaining),floor(2/3*nrow(remaining)))
crossval= remaining[index2, ]
test = remaining[-index2, ]
actual_class=crossval$class
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
class <- as.factor(class)
t.log1 = Sys.time()
logistic.fit <- glm(class~., data = train, family = binomial)
t.log2 = Sys.time()
logistic.probs <- predict(logistic.fit, crossval, type = "response")
logistic.pred <- rep(0, nrow(crossval))
logistic.pred[logistic.probs > 0.5] = 1
actual <- crossval$class
logistic_table <- table(logistic.pred, actual)
f_cfm(logistic_table)

#Decision Tree
library(tree)
t.tree1 = Sys.time()
tree.fit <- tree(class~., data= train)
t.tree2 = Sys.time()
cv.class <- cv.tree(tree.fit, FUN= prune.misclass)
cv.class #dev corresponds to misclassification error rate
par(mfrow=c(1,2))
plot(cv.class$size, cv.class$dev, type= "b")
plot(cv.class$k, cv.class$dev, type= "b")
prune.tree <- prune.misclass(tree.fit, best = 8)
plot(prune.tree)
text(prune.tree, pretty = 0)
tree.pred <- predict(prune.tree, crossval, type = "class")
tree_table <- table(tree.pred, actual)
f_cfm(tree_table)

#Random Forest
library(randomForest)
used_pred <- floor(sqrt(ncol(data)-1))
t.rf1 = Sys.time()
rf.fit <- randomForest(class~., data = train, mtry = used_pred,
                       importance = TRUE, maxdepth = 8)
t.rf2 = Sys.time()
rf.pred <- predict(rf.fit, newdata = crossval)
rf_table <- table(rf.pred, actual)
f_cfm(rf_table)

data2 = data
par(mfrow = c(2,2))
for(i in 1:4){
  hist(data2[,i],freq = F , main  = paste("Histogram of", colnames(data2)[i]))
  lines(density(data2[,i]), col = 'Red')
}

for(i in 1:(ncol(data2)-1)){
  l1 = BoxCox.lambda(data2[,i], method = 'guerrero', lower = -5, upper = 5)
  print(l1)
  data2[,i] = BoxCox(data2[,i], l1)
}

par(mfrow = c(2,2))
for(i in 1:4){
  hist(data2[,i],freq = F , main  = paste("Histogram of", colnames(data2)[i]))
  lines(density(data2[,i]), col = 'Red')
}


#Data Split
names(data2)
index= sample(1:nrow(data2),floor(0.85*nrow(data2)))
train2= data2[index, ]
remaining2= data2[-index, ]
index2 = sample(1:nrow(remaining2),floor(2/3*nrow(remaining2)))
crossval2 = remaining2[index2, ]
test2 = remaining2[-index2, ]
actual2=crossval2$class
dim(train2)
dim(crossval2)
dim(test2)

library(MASS)
t.lda1 = Sys.time()
lda.fit= lda(class~.,data= train2)
t.lda2 = Sys.time()
lda.pred= predict(lda.fit, crossval2, type="response")$class
lda_table <- table(lda.pred, actual2)
f_cfm(lda_table)

t.qda1 = Sys.time()
qda.fit= qda(class~.,data= train2)
t.qda2 = Sys.time()
qda.pred= predict(qda.fit, crossval2, type="response")$class
qda_table <- table(qda.pred, actual2)
f_cfm(qda_table)

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

print("Time taken:")
t.log2-t.log1
t.tree2 - t.tree1
t.rf2 - t.rf1
t.lda2 - t.lda1
t.qda2 - t.qda1

print("F-Scores")
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
logistic_roc = roc(crossval$class ~ logistic_prob, plot = TRUE, main= "ROC Curve for Logistic Regression", print.auc = TRUE)
paste("Area under the curve is",round(auc(logistic_roc), 5))

#LDA
lda_prob= predict(lda.fit, crossval2, type="response")$posterior[,2]
lda_roc = roc(crossval2$class ~ lda_prob, plot = TRUE, main= "ROC Curve for LDA", print.auc = TRUE)
paste("Area under the curve is",round(auc(lda_roc), 5))

#QDA
qda_prob= predict(qda.fit, crossval2, type="response")$posterior[,2]
qda_roc = roc(crossval$class ~ qda_prob, plot = TRUE, main= "ROC Curve for QDA", print.auc = TRUE)
paste("Area under the curve is",round(auc(qda_roc), 5))

#Decision Tree
tree_predict= predict(prune.tree, crossval, type="vector")
tree_roc = roc(crossval$class~tree_predict[,2], plot = TRUE, main= "ROC Curve for Decision Tree", print.auc = TRUE)
paste("Area under the curve is",round(auc(tree_roc), 10))

#Random Forest
rf_predict= predict(rf.fit, crossval, type="prob")
rf_roc = roc(crossval$class~rf_predict[,2], plot = TRUE, main= "ROC Curve for Random Forest", print.auc = TRUE)
paste("Area under the curve is",round(auc(rf_roc), 10))

#Final Fit
final_fit <- predict(rf.fit, newdata = test)
actual_test <- test$class
final_rf_table <- table(final_fit, actual_test)
f_cfm(final_rf_table)

#Evaluation Metric for Test Set

#F_Score
F_score(final_rf_table)

rf_predict_test= predict(rf.fit, test, type="prob")
tree_roc_test = roc(test$class~rf_predict_test[,2], plot = TRUE, main="ROC Curve for Final Model", print.auc = TRUE)
paste("Area under the curve is",round(auc(tree_roc_test),4))


