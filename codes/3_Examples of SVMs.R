
getwd()
# setwd("cw1/")
library(R.matlab)
library(e1071)
library(tidyverse)
library(magick)

pch_types <- c(16, 17)
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

mce <- function(y, yhat){
  return(sum(y!=yhat)/length(y))
}


# Read the dataset -- this dataset is from Andrew Ng's Machine Learning Course
data <- readMat("data/ex6data2.mat")

X <- data$X
y <- data$y
df <- data.frame(cbind(X, y))
names(df)[3] <- "y"

xlims <- c(min(X[,1])-0.05, max(X[,1])+0.05)
ylims <- c(min(X[,2])-0.05, max(X[,2])+0.05)
plot_data(X, y, main="", xlim=xlims, ylim=ylims)

# Fit the data
rbf <- svm(y ~ ., data = df, type="C-classification", kernel="radial", scale=FALSE, gamma=0.1, cost = 5)
gamma

n <- 100
x1_vals <- seq(min(X[, 1]), max(X[, 1]), length.out = n)
x2_vals <- seq(min(X[, 2]), max(X[, 2]), length.out = n)

xgrid <- expand.grid(X1 = x1_vals, X2 = x2_vals)
xgrid <- data.frame(cbind(xgrid$X1, xgrid$X2))

preds <- predict(rbf, xgrid, decision.values=TRUE)

probs <- attributes(preds)$decision.values
ygrid <- as.numeric(as.character(preds))

plot(xgrid, col = c("red", "blue")[ygrid+1], pch=20, cex=0.1)
points(X, col = c("red", "blue")[y+1], pch=pch_types[y+1])

contour(x1_vals, x2_vals, matrix(probs, n, n), level=0, add = TRUE)

## Create a grid to simulate the different decision boundaries 
png("3_Hyperparameters of Radial SVM.png", units="mm", width=400, height=300, res=100)
costs <- gammas <- 10^(seq(-1, 1))
par(mfrow=c(length(costs), length(gammas)), mar=c(3, 2, 3 ,2))
for (cost in costs){
  for (gamma in gammas){
    model <- svm(y ~ ., data=df, type="C-classification",
                 kernel="radial", gamma=gamma, cost=cost, scale=TRUE)
    
    preds <- predict(model, xgrid, decision.values=TRUE)
    probs <- attributes(preds)$decision.values
    ygrid <- as.numeric(as.character(preds))
    
    yhat <- predict(model, df)
    error <- mce(y, yhat)
    
    plot(xgrid, col=c("red", "blue")[ygrid+1], pch=20, cex=0.1, 
         xlab = "", ylab = "",
         yaxt = 'n', xaxt='n',
         main=sprintf("MCE=%s, cost=%s, gamma=%s", round(error, 2), cost, gamma), cex.main=1.4)
    points(X, col=c("red", "blue")[y+1], pch=pch_types[y+1])
    
    contour(x1_vals, x2_vals, matrix(probs, n, n), level=0, add=TRUE)
  }
}

dev.off()
graphics.off()

## Create a GIF to illustrate the effect of the parameter gamma on the decision boundary

# create dir to store images
dir.create("rbf-gamma")
par(mar=c(0.5, 0.5, 0.5, 0.5))
gammas <- seq(10^-1, 10^1, by=0.1)
i <- 1
for (gamma in gammas){
  model <- svm(y ~ ., data=df, type="C-classification",
               kernel="radial", gamma=gamma, cost=cost, scale=TRUE)
  
  preds <- predict(model, xgrid, decision.values=TRUE)
  probs <- attributes(preds)$decision.values
  ygrid <- as.numeric(as.character(preds))
  
  yhat <- predict(model, df)
  error <- mce(y, yhat)
  
  filename <-  sprintf("rbf-gamma/plot_gamma_%03d.png", i)
  
  png(filename, width=200, height=150, units = "mm", res=200)
  plot(xgrid, col=c("red", "blue")[ygrid+1], pch=20, cex=0.1, 
       xlab = "", ylab = "",
       yaxt = 'n', xaxt='n',
       main=sprintf("MCE=%s, cost=%s, gamma=%s", round(error, 2), cost, gamma), cex.main=1)
  points(X, col=c("red", "blue")[y+1], pch=pch_types[y+1])
  
  contour(x1_vals, x2_vals, matrix(probs, n, n), level=0, add=TRUE)
  dev.off()
  i <- i+1
}

filenames <- sprintf("rbf-gamma/plot_gamma_%03d.png", 1:30)
m <- magick::image_read(filenames[1])
for (i in 2:30){
  m <- c(m, magick::image_read(filenames[i]))
} 
m <- magick::image_animate(m, fps = 4, dispose = "previous")
magick::image_write(m, "rbf_and_gamma.gif")

## Polynomial SVM
poly <- svm(y ~ ., data = df, type="C-classification", kernel="polynomial", scale=TRUE,
              degree=1, coef0=3)

preds <- predict(poly, xgrid, decision.values=TRUE)
probs <- attributes(preds)$decision.values
ygrid <- as.numeric(as.character(preds))

plot(xgrid, col = c("red", "blue")[ygrid+1], pch=20, cex=0.1)
points(X, col = c("red", "blue")[y+1], pch=pch_types[y+1])

contour(x1_vals, x2_vals, matrix(probs, n, n), level=0, add = TRUE)

dir.create("poly-coef")

png("polynomial_degree.png", width=200, height=150, units = "mm", res=200)
par(mfrow=c(2, 3), mar=c(1, 1, 2, 1))
degrees <- seq(1:6)
i <- 1
for (degree in degrees){
  model <- svm(y ~ ., data=df, type="C-classification",
               kernel="poly", degree=degree, cost=cost, scale=TRUE)
  
  preds <- predict(model, xgrid, decision.values=TRUE)
  probs <- attributes(preds)$decision.values
  ygrid <- as.numeric(as.character(preds))
  
  yhat <- predict(model, df)
  error <- mce(y, yhat)
  
  filename <-  sprintf("poly-degrees/plot_degree_%03d.png", i)
  
  #png(filename, width=200, height=150, units = "mm", res=200)
  plot(xgrid, col=c("red", "blue")[ygrid+1], pch=20, cex=0.1, 
       xlab = "", ylab = "",
       yaxt = 'n', xaxt='n',
       main=sprintf("MCE=%s, cost=%s, degree=%s", round(error, 2), cost, degree), cex.main=1)
  points(X, col=c("red", "blue")[y+1], pch=pch_types[y+1])
  
  contour(x1_vals, x2_vals, matrix(probs, n, n), level=0, add=TRUE)
  #dev.off()
  i <- i+1
}

dev.off()

png("polynomial_coef.png", width=200, height=150, units = "mm", res=200)
par(mfrow=c(3, 3), mar=c(1, 1, 2, 1))
coefs <- seq(-3, 3, by=1)
i <- 1
for (coef in coefs){
  model <- svm(y ~ ., data=df, type="C-classification",
               kernel="poly", degree=3, coef=coef, cost=10, scale=TRUE)
  
  preds <- predict(model, xgrid, decision.values=TRUE)
  probs <- attributes(preds)$decision.values
  ygrid <- as.numeric(as.character(preds))
  
  yhat <- predict(model, df)
  error <- mce(y, yhat)
  
  filename <-  sprintf("poly-degrees/plot_degree_%03d.png", i)

  #png(filename, width=200, height=150, units = "mm", res=200)
  plot(xgrid, col=c("red", "blue")[ygrid+1], pch=20, cex=0.1, 
       xlab = "", ylab = "",
       yaxt = 'n', xaxt='n',
       main=sprintf("MCE=%s, cost=%s, coef=%s", round(error, 2), cost, coef), cex.main=1)
  points(X, col=c("red", "blue")[y+1], pch=pch_types[y+1])
  
  contour(x1_vals, x2_vals, matrix(probs, n, n), level=0, add=TRUE)
  #dev.off()
  i <- i+1
}

dev.off()








