rm(list=ls())
set.seed(2022)
library(ISLR)
Data <- Caravan
str(Data)
dim(Data)

#Checking for missing values
sum(is.na(data))

#Check for Data Imbalance
attach(Data)
sum(Purchase == "Yes") / nrow(Data)
sum(Purchase == "No") / nrow(Data)
library(DMwR)
Data$Purchase <- as.factor(Purchase)
data <- SMOTE(Purchase~., Data, perc.over= 200, k= 5,  perc.under= 150)
attach(data)
sum(Purchase == "Yes")*100/nrow(data)
sum(Purchase == "No")*100/nrow(data)

#Data Split
names(data)
index= sample(1:nrow(data),floor(0.85*nrow(data)))
train= data[index, ]
remaining= data[-index, ]
index2 = sample(1:nrow(remaining),floor(2/3*nrow(remaining)))
crossval= remaining[index2, ]
test = remaining[-index2, ]
actual_Outcome=crossval$Outcome
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
Purchase <- as.factor(Purchase)
threshold <- seq(0.1, 0.9, 0.01)
fscore <- array(0)
for(i in 1:length(threshold)){
  logistic.fit <- glm(Purchase~., data = train, family = binomial)
  logistic.probs <- predict(logistic.fit, crossval, type = "response")
  logistic.pred <- rep("No", nrow(crossval))
  logistic.pred[logistic.probs > threshold[i]]= "Yes"
  tab <- table(logistic.pred, crossval$Purchase)
  prec <- tab[2,2]/(tab[2,2] + tab[2,1])
  #print(prec)
  recall <- tab[2,2]/(tab[2,2] + tab[1,2])
  #print(recall)
  fscore[i] <- (2*prec*recall)/(prec + recall)
}
data.frame(threshold, fscore)
max_acc <- which.max(fscore)
paste("Maximum F1-score is for thresold value of ", threshold[max_acc], " and is = ",round(fscore[max_acc],4))
t.log1 = Sys.time()
logistic.fit <- glm(Purchase~., data = train, family = binomial)
t.log2 = Sys.time()
logistic.probs <- predict(logistic.fit, crossval, type = "response")
logistic.pred <- rep("No", nrow(crossval))
logistic.pred[logistic.probs > threshold[max_acc]] = "Yes"
actual <- crossval$Purchase
logistic_table <- table(logistic.pred, actual)
f_cfm(logistic_table)

#Normality Test
library(energy)
n <- ncol(data)-1
mvnorm.etest(data[,1:n], R= 100)
n <- ncol(data)-1
library(Hmisc)
hist.data.frame(train[,1:17])
hist.data.frame(train[,18:34])
hist.data.frame(train[,35:51])
hist.data.frame(train[,52:68])
hist.data.frame(train[,69:85])

#Decision Tree
library(tree)
t.tree1 = Sys.time()
tree.fit <- tree(Purchase~., data= train)
t.tree2 = Sys.time()
cv.purchase <- cv.tree(tree.fit, FUN= prune.misclass)
cv.purchase #dev corresponds to misclassification error rate
par(mfrow=c(1,2))
plot(cv.purchase$size, cv.purchase$dev, type= "b")
plot(cv.purchase$k, cv.purchase$dev, type= "b")
prune.tree <- prune.misclass(tree.fit, best = 11)
plot(prune.tree)
text(prune.tree, pretty = 0)
tree.pred <- predict(prune.tree, crossval, type = "class")
tree_table <- table(tree.pred, actual)
f_cfm(tree_table)

#Random Forest
library(randomForest)
used_pred <- floor(sqrt(ncol(data)-1))
t.rf1 = Sys.time()
rf.fit <- randomForest(Purchase~., data = train, mtry = used_pred,
                       importance = TRUE, maxdepth = 8)
t.rf2 = Sys.time()
rf.pred <- predict(rf.fit, newdata = crossval)
rf_table <- table(rf.pred, actual)
f_cfm(rf_table)

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


print("F-Scores")
F_score(logistic_table)
F_score(tree_table)
F_score(rf_table)
paste("We have got the maximum F-Score for Random Forest which is", round(F_score(rf_table), 4))

#Plotting ROC curve

#Logistic Regression
library(pROC)
logistic_prob = predict(logistic.fit, newdata = crossval, type = "response")
logistic_roc = roc(crossval$Purchase ~ logistic_prob, plot = TRUE, print.auc = TRUE, main ="ROC Curve for Logistic Regression")
paste("Area under the curve is",round(auc(logistic_roc), 4))

#Decision Tree
tree_predict= predict(prune.tree, crossval, type="vector")
tree_roc = roc(crossval$Purchase~tree_predict[,2], plot = TRUE, print.auc = TRUE, main ="ROC Curve for Decision Tree")
paste("Area under the curve is",round(auc(tree_roc), 4))

#Random Forest
rf_predict= predict(rf.fit, crossval, type="prob")
rf_roc = roc(crossval$Purchase~rf_predict[,2], plot = TRUE, print.auc = TRUE, main ="ROC Curve for Random Forest")
paste("Area under the curve is",round(auc(rf_roc), 4))

#Final Fit

final_fit <- predict(logistic.fit, test, type = "response")
final.pred <- rep("No", nrow(test))
final.pred[final_fit> threshold[max_acc]] = "Yes"
actual <- test$Purchase
final_table <- table(final.pred, actual)
f_cfm(final_table)

#Evaluation Metric for Test Set

F_score(final_table)
final_roc = roc(test$Purchase ~ final_fit, plot = TRUE, print.auc = TRUE, main ="ROC Curve for Final Model")
paste("Area under the curve is",round(auc(final_roc), 4))


