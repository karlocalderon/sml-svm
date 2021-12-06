
#' -----
#' Linear Classifiers on Linearly Separable Data
#' Author: kjcalderon
#' ...
#' 
#' This notebook investigate how the linear classifiers behave on perfectly 
#' linearly separable data. The classifiers that will be explored will are:
#' (1) Logistic Regression, (2) LDA, and (3) SVM with a linear kernel
#' -----

# Packages
library(e1071)
library(MASS)

# Functions
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

#### 1. Simulate data -----

# Load data - we will make use of the classifcationExample dataset from the notes,
# modified by removing the support vectors of a vanilla SVM with a linear kernel 
# to create data that can be separated by a linear boundary
load("../data/classificationExample.RData")
plot_data(X, y, main="")

# Fit an SVM to create a linear decision boundary
svm <- svm(y ~ X, kernel="linear", type="C-classification", cost=1000)

# Get the support vectors
idx <- svm$index
sv <- X[idx, ]
points(sv, pch=pch_types[y+1], col=6)

# Remove the support vectors 
X <- X[-idx,]
y <- y[-idx,]
plot_data(X, y, main="")

rm(sv, svm, idx)

#### The models ----
par(mfrow=c(1, 3))

#### 2. Logistic Regression ----

logit <- glm(y ~ X, family=binomial(link="logit"), control=list(maxit=30))
logit_coef <- coef(logit)

plot_data(X, y, main="Logistic Regression")
abline(-logit_coef[1]/logit_coef[3], -logit_coef[2]/logit_coef[3], col="black")

#### 3. LDA ----

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

plot_data(X, y, main="LDA")
abline(-1*lda$theta_null/lda$theta[2], -1*lda$theta[1]/lda$theta[2], col="black")

#### 4. Linear SVM ----

obj <- tune.svm(
  x = X, y = as.factor(y),
  kernel = "linear", 
  type = "C-classification",
  scale = FALSE,
  cost = 1:50
)

# The best model is at cost = 1
obj$best.model

# Fit the model and plot
svm <- svm(y ~ X, type="C-classification", kernel="linear", cost=1, scale=FALSE)
svm_coef <- coef(svm)

plot_data(X, y, main="Linear SVM, cost=1")
abline(-1*svm_coef[[1]]/svm_coef[[3]], -1*svm_coef[[2]]/svm_coef[[3]], col="black")
par(mfrow=c(1, 1))

### Separable data from MVN ----

# Generate data form multivariate normal

set.seed(42)
mu_x <- c(-1, 3)
mu_o <- c(1, -4)
S <- matrix(c(0.3, 0.05, 0.05, 1.7), nrow=2, ncol=2)

X_x <- mvtnorm::rmvnorm(n=100, mean=mu_x, sigma=S)
X_o <- mvtnorm::rmvnorm(n=100, mean=mu_o, sigma=S)

Xnorm <- rbind(X_x, X_o)
ynorm <- c(rep(1, 100), rep(0, 100))

plot_data(Xnorm, ynorm, main="", xlims=c(-4, 4), ylims=c(-8, 8))


#### Run models again ----

par(mfrow=c(1, 3))

## Logistic
logit <- glm(ynorm ~ Xnorm, family=binomial(link="logit"), control=list(maxit=30))
logit_coef <- coef(logit)

plot_data(Xnorm, ynorm, main="Logistic Regression", xlims=c(-4, 4), ylims=c(-8, 8))
abline(-logit_coef[1]/logit_coef[3], -logit_coef[2]/logit_coef[3], col="black")

## LDA
lda <- fit_lda(Xnorm, ynorm)

plot_data(Xnorm, ynorm, main="LDA", xlims=c(-4, 4), ylims=c(-8, 8))
abline(-1*lda$theta_null/lda$theta[2], -1*lda$theta[1]/lda$theta[2], col="black")

## SVM 

obj <- tune.svm(
  x = Xnorm, y = as.factor(ynorm),
  kernel = "linear", 
  type = "C-classification",
  scale = FALSE,
  cost = 1:50
)

# The best model is at cost = 1
obj$best.model

# Fit the model and plot
svm <- svm(ynorm ~ Xnorm, type="C-classification", kernel="linear", cost=1, scale=FALSE)
svm_coef <- coef(svm)

plot_data(Xnorm, ynorm, main="Linear SVM, cost=1", xlims=c(-4, 4), ylims=c(-8, 8))
abline(-1*svm_coef[[1]]/svm_coef[[3]], -1*svm_coef[[2]]/svm_coef[[3]], col="black")
par(mfrow=c(1, 1))




















