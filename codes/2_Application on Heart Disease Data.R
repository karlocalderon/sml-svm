
#' -----
#' Application of SVM to a real dataset
#' Author: kjcalderon
#' ...
#' 
#' This notebook applies SVM to a real dataset
#' -----

# Packages
library(curl)
library(caret)
library(ROCR)
library(MASS)
library(e1071)

# Load data
URL <- url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
df <- read.csv(URL, header=FALSE)

# Columns 
cols <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", 
          "slope", "ca", "thal", "num")

names(df) <- cols

# Convert nums column to binary response (0/1)
df$num <- ifelse(df$num>0, 1, 0)
df$num <- as.factor(df$num)
table(df$num)

# Convert char data to numeric
df$ca <- as.numeric(df$ca)
df$thal <- as.numeric(df$thal)

df$sex <- as.factor(df$sex)
df$cp <- as.factor(df$cp)
df$fbs <- as.factor(df$fbs)
df$restecg <- as.factor(df$restecg)
df$exang <- as.factor(df$exang)
df$slope <- as.factor(df$slope)
df$thal <- as.factor(df$thal)

# Check for missing
lapply(df, function(x) sum(is.na(x)))
df <- na.omit(df)

# Split data
set.seed(42)
idx <- createDataPartition(df$num, p=0.7, list=FALSE)

train <- df[idx,]
test <- df[-idx,]

# Response rates
table(train$num)/nrow(train)
table(test$num)/nrow(test)

# Create function to measure performance
mce <- function(y, yhat){
  return(sum(y!=yhat)/length(y))
}

# Logistic
logit <- glm(num ~ ., data=train, family=binomial(link="logit"))
summary(logit)

y <- as.numeric(as.character(train$num))
logodds <- predict(logit, newdata=train)
yhat <- ifelse(logodds > 0, 1, 0)

cm <- confusionMatrix(table(y, yhat)) # 0.875

# LDA 
lda <- MASS::lda(num ~ ., data=train)
yhat <- as.numeric(as.character(predict(lda, newdata=train)$class))

cm <- confusionMatrix(table(y, yhat)) # 0.8606

# SVM 
svm <- svm(num ~ ., data=train, type="C-classification", kernel="linear")
yhat <- predict(svm, train)

cm <- confusionMatrix(table(y, yhat)) # 0.8798

# Create folds
set.seed(42)
K <- 5
train$kfold <- createFolds(train$num, k=K, list = FALSE)

# Range of values of cost
costs <- seq(0.001, 1, by=0.001)
N <- length(costs)
results <- matrix(NA, nrow=length(costs), ncol=3)
for (i in 1:N){
  cost <- costs[i]
  train_mces <- c()
  test_mces <- c()
  #mces <- c()
  for (k in 1:K){
    
    kfold_train <- train[train$kfold!=k, which(names(train)!="kfold")]
    kfold_test <- train[train$kfold==k, which(names(train)!="kfold")]
    svm <- svm(num ~ ., data=kfold_train, cost=cost, type="C-classification", kernel="linear")
    
    # Apply to train set
    yhat <- predict(svm, newdata=kfold_train)
    y <- kfold_train$num
    
    train_mces <- c(train_mces, mce(y, yhat))
    
    # Apply to test set
    yhat <- predict(svm, newdata=kfold_test)
    y <- kfold_test$num
    
    test_mces <- c(test_mces, mce(y, yhat))
  }
  
  results[i, 1] <- cost
  results[i, 2] <- mean(train_mces)
  results[i, 3] <- mean(test_mces)
}

plot(results[, 1], results[, 2], col="blue", type="l", ylim=c(0.10, 0.2))
lines(results[, 1], results[, 3], col="red")

results[which.min(results[, 3]),]

train$kfold <- NULL
svm <- svm <- svm(num ~ ., data=train, cost=0.459, type="C-classification", kernel="linear")

yhat <- predict(svm, train)
y <- train$num

cm <- confusionMatrix(table(y, yhat)) # 0.8702

## Assess to test data
y <- test$num

logodds <- predict(logit, newdata=test)
test_logit <- ifelse(logodds > 0, 1, 0)
mce(y, test_logit)
confusionMatrix(table(y, test_logit))

test_lda <- predict(lda, newdata=test)$class
mce(y, test_lda)
confusionMatrix(table(y, test_lda))

test_svm <- predict(svm, newdata=test)
mce(y, test_svm)
confusionMatrix(table(y, test_svm))




