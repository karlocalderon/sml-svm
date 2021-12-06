
#' -----
#' Linear Classifiers on data from the MVN
#' Author: kjcaldeon
#' ...
#' 
#' This notebook investigate how the linear classifiers behave when the
#' data is generated from a MVN with the same variance-covariance matrix
#' (1) Logistic Regression, (2) LDA, and (3) SVM with a linear kernel
#' -----

## Packages
library(e1071)
library(mvtnorm)
library(pROC)

options(scipen=9)

## Functions
pch_types <- c(1, 4)
plot_data <- function(X, y, main, cex.main=1, cex.axis=1, 
                      xlims=c(-5, 28), ylims=c(-2, 20)){
  #' A function to plot data
  plot(X, xlim=xlims, ylim=ylims, 
       xlab="x_1", ylab="x_2", 
       pch=pch_types[y+1],
       col=c("red", "blue")[y+1],
       main=main,
       cex.main=cex.main, 
       cex.axis=cex.axis)
}

odds_to_prob <- function(x){
  return(exp(x)/(1+exp(x)))
}

mce <- function(y, yhat){
  return(sum(y!=yhat)/length(y))
}

metrics <- function(y, yhat){
  
  # Create confusion matrix
  cm <- table(y, yhat)
  
  # Unravel
  tp <- cm[2, 2]
  tn <- cm[1, 1]
  fp <- cm[1, 2]
  fn <- cm[2, 1]
  
  # Metrics 
  acc <- sum(y==yhat)/length(y)
  aucroc <- as.numeric(auc(y, yhat))
  prec <- tp/(tp+fp)
  tpr <- tp/(tp + fn) 
  tnr  <- tn/(tn+fp)
  f1 <- (2*prec*tpr)/(prec+tpr)
  
  return(list(accuracy=acc, 
              auc = aucroc, 
              precision=prec,
              recall=tpr,
              sensitivity=tnr, 
              f1=f1))
}

fit_lda <- function(X, y, prior=c(0.5, 0.5)){
  
  #' A function to calculate the coefficient of the decision boundary of an LDA
  #' ...
  #' 
  #' Params:
  #' -----
  #' X : a matrix of the feature space
  #' y : a vector of the true labels of X
  #' prior : a vector of the prior probabilities assigned to class o, x respectively
  #' 
  #' Output:
  #' -----
  #' theta_null : a value, the intercept of the LDA decision boundary
  #' theta : a vector of the coefficients of the Xs in the LDA decision boundary
  #' 
  #' Note:
  #' ----
  #' LDA can be implemented using the function lda from {MASS} but I could not find a way to extract
  #' the coefficients of the decision boundary from the results of the package;
  #' hence reverting to the base R implementation from the notes
  
  # N, mean, and variance
  N_o <- sum(y==0)
  N_x <- sum(y==1)
  mean_o <- colMeans(X[y==0,])
  mean_x <- colMeans(X[y==1,])
  S_o <- cov(X[y==0,])
  S_x <- cov(X[y==1,])
  
  # Calculate pooled variance
  pi_o <- prior[1]
  pi_x <- prior[2]
  n <- dim(X)[1]
  G <- length(unique(y))
  
  S_pooled <- (1/(n-G)) * ((N_o * S_o) + (N_x * S_x))
  
  # Invert variance
  S_inv <- solve(S_pooled)
  
  theta_null <- (-1/2)*((t(mean_o) %*% S_inv %*% mean_o) - (t(mean_x) %*% S_inv %*% mean_x)) + log(pi_o/pi_x)
  theta <- S_inv %*% (mean_o - mean_x)
  
  return(list(theta_null=theta_null, theta=theta))
}

lda <- fit_lda(X, y)

## Specify the parameters of the multivariate normal distribution

# Generate random means
set.seed(123)
mu_x <- runif(2, -3, 1)

set.seed(421321) 
mu_o <- runif(2, -2, 0)

# Generate random S, and S should be positive definite
set.seed(213565)
S <- matrix(c(0.3, 0.05, 0.05, 1.7), nrow=2, ncol=2)

# First we generate test data
set.seed(42)
n <- 100
X_x <- mvtnorm::rmvnorm(n=n, mean=mu_x, sigma=S)
X_o <- mvtnorm::rmvnorm(n=n, mean=mu_o, sigma=S)

test_X <- rbind(X_x, X_o)
colnames(test_X) <- c("X1", "X2")
test_y <- c(rep(0, n), rep(1, n))

rm(X_x, X_o)

df_test <- data.frame(test_X)

plot_data(test_X, test_y, main="", xlims=c(-4, 1), ylims=c(-4, 4))

# Now we create train 

results <- matrix(data=NA, nrow=100, ncol=11)
for (i in 1:100){
  
  # Generate data
  s <- .Random.seed[i]
  set.seed(s)

  results[i, 1] <- s
  n <- 1000
  X_x <- mvtnorm::rmvnorm(n=n, mean=mu_x, sigma=S)
  X_o <- mvtnorm::rmvnorm(n=n, mean=mu_o, sigma=S)
  
  train_X <- rbind(X_x, X_o)
  colnames(train_X) <- c("X1", "X2")
  train_y <- c(rep(0, n), rep(1, n))
  rm(X_x, X_o)
  
  df_train <- data.frame(cbind(train_X, train_y))
  
  # Logistic Regression
  logit <- glm(train_y ~ ., data=df_train, family=binomial(link="logit"))
  log_odds <- predict(logit, newdata=df_test, type = "link")
  probs <- odds_to_prob(log_odds)
  preds <- ifelse(probs>0.5, 1, 0)
  
  results[i, 2] <- mce(test_y, preds)
  results[i, 3] <- metrics(test_y, preds)$accuracy
  results[i, 4] <- metrics(test_y, preds)$auc
  
  # LDA
  lda <- fit_lda(train_X, train_y)
  probs <- (test_X %*% lda$theta)
  probs <- probs + rep(lda$theta_null, length(test_y))
  preds <- ifelse(probs<0, 1, 0)

  results[i, 5] <- mce(test_y, preds)
  results[i, 6] <- metrics(test_y, preds)$accuracy
  results[i, 7] <- metrics(test_y, preds)$auc
  
  # SVM 
  # obj <- tune.svm(
  #   x = train_X, y = as.factor(train_y),
  #   kernel = "linear", 
  #   type = "C-classification",
  #   scale = FALSE,
  #   cost = c(1, 10, 20, 30, 40, 50),
  #   tunecontrol = tune.control(cross=5)
  # )
  
  # svm <- obj$best.model
  svm <- e1071::svm(train_y ~ ., data=df_train, type="C-classification", kernel="linear")
  preds <- predict(svm, newdata=as.data.frame(test_X), type="response")
  preds <- as.numeric(as.character(preds))
  
  results[i, 8] <- svm$cost
  results[i, 9] <- mce(test_y, preds)
  results[i, 10] <- metrics(test_y, preds)$accuracy
  results[i, 11] <- metrics(test_y, preds)$auc

}

# Convert to dataframe 
results <- data.frame(results)
names(results) <- c("seed", 
                    "logit_mce", "logit_acc", "logit_auc",
                    "lda_mce", "lda_acc", "lda_auc",
                    "svm_cost",
                    "svm_mce", "svm_acc", "svm_auc")

colMeans(results)
saveRDS(results, "02_MVN-Results.RDS")







