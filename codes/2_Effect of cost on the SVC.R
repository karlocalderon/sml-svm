  
  library(curl)
  library(tidyverse)
  
  options(scipen=99)
  
  # Load data
  # URL <- url("https://hastie.su.domains/ElemStatLearn/datasets/ESL.mixture.rda")
  # load(URL)
  
  load("classificationExample.RData")
  
  X[y==0, 1] <- X[y==0, 1] + 2.5
  
  #X <- ESL.mixture$x
  #y <- ESL.mixture$y
  
  # set.seed(6435)
  # mu_x <- runif(2, -2.0, 0.5)
  # 
  # set.seed(43256)
  # mu_o <- runif(2, -1.75, 1.5)
  # 
  # # Generate random S, and S should be positive definite
  # set.seed(62343)
  # Sx <- matrix(c(4.2, 1.5, 1.5, 3.7), nrow=2, ncol=2)
  # So <- matrix(c(3.9, 1.3, 1.3, 4.1), nrow=2, ncol=2)
  # # First we generate test data
  # set.seed(42)
  # n <- 500
  # X_x <- mvtnorm::rmvnorm(n=n, mean=mu_x, sigma=Sx)
  # X_o <- mvtnorm::rmvnorm(n=n, mean=mu_o, sigma=So)
  # 
  # X <- rbind(X_o, X_x)
  # y <- c(rep(0, n), rep(1, n))
  
  # Functions 
  pch_types <- c(1, 4)
  plot_data <- function(X, y){
    xlims <- c(floor(min(X[, 1]))-1, ceiling(max(X[, 1]))+1)
    ylims <- c(floor(min(X[, 2]))-1, ceiling(max(X[, 2]))+1)
    plot(X, 
         xlim=xlims, 
         ylim=ylims,
         xlab="",
         ylab="",# "x_2",
         pch=pch_types[y+1],
         col=alpha(c("red", "blue")[y+1], 0.2), 
         cex=0.75,
         cex.axis=0.8,
         mar=c(0, 0, 0, 0))
  }
  
  mce <- function(y, yhat){
    #' A function to calculate the misclassification error 
    return(sum(y!=yhat)/length(y))
  }
  
  # Plot mixture dataset
  #png("plot.png", width=200, height=125, units="mm", res=200)
  
  #### Base plot ----
  
  # fit a linear SVM
  svm <- e1071::svm(y ~ X, type="C-classification", kernel="linear", cost=10, scale=FALSE)
  beta <- coef(svm)
  
  # Plot data
  plot_data(X, y)
  
  # plot decision boundary
  abline(-beta[[1]]/beta[[3]], -beta[[2]]/beta[[3]], col="black")
  
  # plot margins 1/||w||
  margin <- 1/sqrt(sum(beta[2:3]^2))
  abline((1-beta[[1]])/beta[[3]], -beta[[2]]/beta[[3]], col="black", lty=2, )
  abline((-1-beta[[1]])/beta[[3]], -beta[[2]]/beta[[3]], col="black", lty=2)
  
  # get support vectors
  sv <- svm$index
  length(sv)
  points(X[sv,], pch=pch_types[y[sv]+1], col=c("red", "blue")[y[sv]+1], cex=0.75)# 6, cex=0.75)
  title(
    main=sprintf("SVM with Linear Kernel, cost=%s", as.character(svm$cost)),
    #title(sub=sprintf(" support vectors: %s", length(sv)), cex.sub=.6),
    cex.main = 0.8
  )
  
  legend("topright", sprintf("No. of support vectors: %s", as.character(length(sv))), cex=0.5)
  
  ### Split data ----
  
  n <- dim(X)[1]
  
  set.seed(42)
  idx <- sample(1:n, size=floor(0.2*n))
  
  # Split data
  test_X <- data.frame(X[idx,])
  test_y <- y[idx]
  
  train_X <- data.frame(X[-idx,])
  train_y <- y[-idx]
  
  df_train <- cbind(train_X, y=train_y)
  df_test <- cbind(test_X, y=test_y)
  
  # Generate plot for report
  par(mfrow=c(2,2))
  costs <- c(0.001, 0.01, 0.05, 1)
  for (cost in costs){
    # fit a linear SVM
    svm <- e1071::svm(y ~ ., data=df_train, type="C-classification", kernel="linear", cost=cost, scale=FALSE)
    beta <- coef(svm)
    
    # Plot data
    plot_data(train_X, train_y)
    
    # plot decision boundary
    abline(-beta[[1]]/beta[[3]], -beta[[2]]/beta[[3]], col="black")
    
    # plot margins 1/||w||
    abline((1-beta[[1]])/beta[[3]], -beta[[2]]/beta[[3]], col="black", lty=2, )
    abline((-1-beta[[1]])/beta[[3]], -beta[[2]]/beta[[3]], col="black", lty=2)
    
    # get support vectors
    sv <- svm$index
    length(sv)
    
    # plot support vectors
    points(train_X[sv,], 
           pch=pch_types[train_y[sv]+1], 
           col=c("red", "blue")[train_y[sv]+1], 
           cex=0.75)
    
    title(
      main=sprintf("cost = %s, no. support vectors: %s", as.character(svm$cost), as.character(length(sv))),
      cex.main = 0.8
    )
    
    # get training predictions and MCE
    preds_train <- as.numeric(as.character(svm$fitted))
    train_mce <- round(mce(train_y, preds_train), 3)
    
    # get test predictions and MCE
    preds_test <- as.numeric(as.character(predict(svm, newdata=df_test)))
    test_mce <- round(mce(test_y, preds_test), 3)
    
    text(-4.2, -3, 
         sprintf("Train MCE: %s\nTest MCE: %s", train_mce, test_mce), 
         adj = c(0,0),
         cex = 0.8)
    
  }

# fit loss curves over different values of cost

cost_vals <- seq(0.5, 100, by=0.5)
results <- matrix(NA, nrow=length(cost_vals), ncol=3)
for (i in 1:length(cost_vals)){
  svm <- e1071::svm(y ~ ., 
                    data=df_train, 
                    type="C-classification", 
                    kernel="linear", 
                    cost=cost_vals[i], 
                    scale=FALSE)
  
  preds_train <- as.numeric(as.character(svm$fitted))
  train_mce <- round(mce(train_y, preds_train), 5)
  
  preds_test <- as.numeric(as.character(predict(svm, newdata=df_test)))
  test_mce <- round(mce(test_y, preds_test), 5)
  
  results[i, 1] <- cost_vals[i]
  results[i, 2] <- train_mce
  results[i, 3] <- test_mce
}

par(mfrow=c(1,1))
plot(results[,1], results[, 2], type="l", ylim=c(min(results[, 2:3])-0.05, max(results[, 2:3])-0.05))
lines(results[,1], results[, 3], type="l", col="red")


