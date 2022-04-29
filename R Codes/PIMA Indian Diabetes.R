#Loading the Data

rm(list=ls())
set.seed(1234)
Data<- read.csv("https://raw.githubusercontent.com/rajdeepsaha0809/MTH514A-Multivariate-Project/main/diabetes.csv")
head(Data)
dim(Data)
str(Data)

#Checking for missing values
sum(is.na(Data))

imputer <- function(x){
  med <- median(x[x!=0])
  for(i in 1:length(x)){
    if(x[i] == 0){
      x[i] = med
    }
  }
  return(x)
}


imp = apply(Data[,2:6], 2, imputer)
Data = cbind(Pregnancies=Data$Pregnancies, imp, Data[, 7:ncol(Data)])


library(forecast)


#Check for Data Imbalance
attach(Data)
sum(Outcome == 1) / nrow(Data)
sum(Outcome == 0) / nrow(Data)
library(DMwR)
Data$Outcome <- as.factor(Outcome)
data <- SMOTE(Outcome~., Data, perc.over= 200, k= 5,  perc.under= 150)
attach(data)
sum(Outcome == 1)*100/nrow(data)
sum(Outcome == 0)*100/nrow(data)

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
Outcome <- as.factor(Outcome)
threshold <- seq(0.1, 0.9, 0.01)
fscore <- array(0)
for(i in 1:length(threshold)){
  logistic.fit <- glm(Outcome~., data = train, family = binomial)
  logistic.probs <- predict(logistic.fit, crossval, type = "response")
  logistic.pred <- rep(0, nrow(crossval))
  logistic.pred[logistic.probs > threshold[i]]= 1
  tab <- table(logistic.pred, crossval$Outcome)
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
logistic.fit <- glm(Outcome~., data = train, family = binomial)
t.log2 = Sys.time()
logistic.probs <- predict(logistic.fit, crossval, type = "response")
logistic.pred <- rep(0, nrow(crossval))
logistic.pred[logistic.probs > threshold[max_acc]] = 1
actual <- crossval$Outcome
logistic_table <- table(logistic.pred, actual)
f_cfm(logistic_table)



#Decision Tree
library(tree)
t.tree1 = Sys.time()
tree.fit <- tree(Outcome~., data= train)
t.tree2 = Sys.time()
cv.outcome <- cv.tree(tree.fit, FUN= prune.misclass)
cv.outcome #dev corresponds to misclassification error rate
par(mfrow=c(1,2))
plot(cv.outcome$size, cv.outcome$dev, type= "b")
plot(cv.outcome$k, cv.outcome$dev, type= "b")
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
rf.fit <- randomForest(Outcome~., data = train, mtry = used_pred,
                       importance = TRUE, maxdepth = 8)
t.rf2 = Sys.time()
rf.pred <- predict(rf.fit, newdata = crossval)
rf_table <- table(rf.pred, actual)
f_cfm(rf_table)


data2 = data[,2:ncol(data)]
par(mfrow = c(3,3))
for(i in 1:7){
  hist(data2[,i],freq = F , main  = paste("Histogram of", colnames(data2)[i]))
  lines(density(data2[,i]), col = 'Red')
}

for(i in 1:(ncol(data2)-1)){
  l1 = BoxCox.lambda(data2[,i])
  print(l1)
  data2[,i] = BoxCox(data2[,i], l1)
}

par(mfrow = c(3,3))
for(i in 1:7){
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
actual2=crossval2$Outcome
dim(train2)
dim(crossval2)
dim(test2)

library(MASS)
t.lda1 = Sys.time()
lda.fit= lda(Outcome~.,data= train2)
t.lda2 = Sys.time()
lda.pred= predict(lda.fit, crossval2, type="response")$class
lda_table <- table(lda.pred, actual2)
f_cfm(lda_table)

t.qda1 = Sys.time()
qda.fit= qda(Outcome~.,data= train2)
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
logistic_roc = roc(crossval$Outcome ~ logistic_prob, plot = TRUE, main= "ROC Curve for Logistic Regression", print.auc = TRUE)
paste("Area under the curve is",round(auc(logistic_roc), 5))

#LDA
lda_prob= predict(lda.fit, crossval2, type="response")$posterior[,2]
lda_roc = roc(crossval2$Outcome ~ lda_prob, plot = TRUE, main= "ROC Curve for LDA", print.auc = TRUE)
paste("Area under the curve is",round(auc(lda_roc), 5))

#QDA
qda_prob= predict(qda.fit, crossval2, type="response")$posterior[,2]
qda_roc = roc(crossval$Outcome ~ qda_prob, plot = TRUE, main= "ROC Curve for QDA", print.auc = TRUE)
paste("Area under the curve is",round(auc(qda_roc), 5))

#Decision Tree
tree_predict= predict(prune.tree, crossval, type="vector")
tree_roc = roc(crossval$Outcome~tree_predict[,2], plot = TRUE, main= "ROC Curve for Decision Tree", print.auc = TRUE)
paste("Area under the curve is",round(auc(tree_roc), 10))

#Random Forest
rf_predict= predict(rf.fit, crossval, type="prob")
rf_roc = roc(crossval$Outcome~rf_predict[,2], plot = TRUE, main= "ROC Curve for Random Forest", print.auc = TRUE)
paste("Area under the curve is",round(auc(rf_roc), 10))

#Final Fit
final_fit <- predict(rf.fit, newdata = test)
actual_test <- test$Outcome
final_rf_table <- table(final_fit, actual_test)
f_cfm(final_rf_table)

#Evaluation Metric for Test Set

#F_Score
F_score(final_rf_table)

rf_predict_test= predict(rf.fit, test, type="prob")
tree_roc_test = roc(test$Outcome~rf_predict_test[,2], plot = TRUE, print.auc = TRUE, main= "ROC Curve for Final Model")
paste("Area under the curve is",round(auc(tree_roc_test),4))


