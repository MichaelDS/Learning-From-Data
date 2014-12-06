## Performs the non-linear transformation phi(x1, x2) = (1, x1, x2, x1*x2, x1^2, x2^2)
## Returns the transformed input 
## D - Training set with two-dimensional input
transform.phi <- function(D) {
  x1 <- D[1]
  x2 <- D[2]
  phi <- cbind(1, x1, x2, x1*x2, x1^2, x2^2)
  phi
}

## Performs classification using linear regression and, if specified, uses a non-linear transformation and regularization
## Returns measures of the in-sample and out-of-sample errors achieved
## D_train - Training set with column names
## D_test - Test set with column names
## transform - A function which returns a non-linear transformation on the input of the training set
## lambda - Strength of regularization
## plotBoundary - When set to TRUE, will plot the training and tests sets against the decision boundary; setting this to TRUE will increase the running time of the experiment
digits.regression <- function(train, test, digits = c(1, 5), transform = NULL, lambda = 0, plotBoundary = FALSE) {
  if(length(digits) != 1 && length(digits) != 2)
    stop('Invalid length of digits vector.  Must specify one or two digits to classify')
  if(length(digits) == 2) {
    train <- train[(train$digit == digits[1]) | (train$digit == digits[2]), ]
    test <- test[(test$digit == digits[1]) | (test$digit == digits[2]), ]
  }
  
  train$y <- -1
  test$y <- -1
  train[train$digit == digits[1], ]$y <- 1
  test[test$digit == digits[1], ]$y <- 1
  D_train <- train[, -1]
  D_test <- test[, -1]
  y_train <- D_train[[length(D_train)]] 
  y_test <- D_test[[length(D_test)]]
  if(is.null(transform)) {               # if no transformation is specified, simply use the provided input
    X <- as.matrix(cbind(1, D_train[1:length(D_train)-1]))
    X_test <- as.matrix(cbind(1, D_test[1:length(D_test)-1]))
  }
  else {                                 # otherwise, perform the non-linear transformation
    X <- as.matrix(transform(D_train))
    X_test <- as.matrix(transform(D_test))
  }
  
  w <- solve(t(X)%*%X + lambda*diag(nrow(t(X)%*%X)))%*%t(X)%*%y_train  # least-squares solution via pseudo-inverse for w; one-step learning with regularization
  
  # Apply the final hypothesis to the inputs and calculate E_in and E_out
  y_trainFit <- sign(t(w)%*%t(X))
  E_in <- sum(y_trainFit != y_train)/length(y_trainFit)
  
  y_testFit <- sign(t(w)%*%t(X_test))
  E_out <- sum(y_testFit != y_test)/length(y_testFit)
  
  if(plotBoundary) {
    library(ggplot2)
    library(gridExtra)
    titlePiece <- as.character(lambda)                                # store value of lambda for plot title
    gridRange <- apply(D_train[c('intensity', 'symmetry')], 2, range)
    x1 <- seq(gridRange[1, 1] - 0.25, gridRange[2, 1] + 0.25, .01)
    x2 <- seq(gridRange[1, 2] - 0.5, gridRange[2, 2] + 0.5, .01)
    grid.fit <- expand.grid(list(intensity = x1, symmetry = x2))      # create a data frame of all combinations of specified x1 and x2 values
    if(is.null(transform))
      grid.fit <- cbind(1, grid.fit)
    grid.transformed <- transform(grid.fit)                           # just renames grid.fit if transform is null; otherise, transforms the grid into the feature space defined by the non-linear transformation
    y <- sign(t(w)%*%t(grid.transformed))                             # apply the final hypothesis to every point on the transformed grid in order to obtain predicted y values 
    grid.fit$y <- t(y)                                                # append the predictions to the un-transformed grid
    
    ## Set up the basic plot and the decision boundary
    base <- ggplot(data = grid.fit, aes(intensity, symmetry, fill = as.factor(y))) + geom_tile() +
      xlab("intensity") + ylab("symmetry") + 
      scale_fill_discrete(limits = c(-1, 1)) +                                # was originally using scale_fill_gradient; discrete makes more sense here
      scale_fill_manual(values = c('gray97', 'lightgoldenrod')) +
      labs(fill = 'Decision Boundary')
    
    ## Plot the training set against the base plot
    plot1 <- base + ggtitle(bquote('Training Data: '~lambda == .(titlePiece))) +
      geom_point(data = D_train, aes(intensity, symmetry, colour = as.factor(y))) +
      labs(colour = 'y') + 
      scale_colour_manual(values = c('red', 'blue')) +
      scale_x_continuous(expand = c(0,0)) +
      scale_y_continuous(expand = c(0,0))
    
    ## Plot the test set against the base plot
    plot2 <- base + ggtitle(bquote('Test Data: '~lambda == .(titlePiece))) +
      geom_point(data = D_test, aes(intensity, symmetry, colour = as.factor(y))) +
      labs(colour = 'y') + 
      scale_colour_manual(values = c('red', 'blue')) +
      scale_x_continuous(expand = c(0,0)) +
      scale_y_continuous(expand = c(0,0))
    
    # Print the two plots and add a main title
    grid.arrange(plot1, plot2, ncol = 2, main = textGrob(paste(ifelse(is.null(transform),'Ridge Regression without Non-Linear Transformation\n', 'Ridge Regression with Non-Linear Transformation\n'), digits[1], 'vs', ifelse(length(digits) == 2, digits[2], 'All')), gp=gpar(cex=1.5), vjust = 0.7)) 
  }  
  
  list(E_in = E_in, E_out = E_out) # return E_in and E_out
}

train <- read.table('train.txt', col.names = c('digit', 'intensity', 'symmetry'))
test <- read.table('test.txt', col.names = c('digit', 'intensity', 'symmetry'))

## Problems 7, 8, and 9
digits.regression(train, test, digits = c(0), lambda = 1)
digits.regression(train, test, digits = c(1), lambda = 1)
digits.regression(train, test, digits = c(2), lambda = 1)
digits.regression(train, test, digits = c(3), lambda = 1)
digits.regression(train, test, digits = c(4), lambda = 1)
digits.regression(train, test, digits = c(5), lambda = 1)
digits.regression(train, test, digits = c(6), lambda = 1)
digits.regression(train, test, digits = c(7), lambda = 1)
digits.regression(train, test, digits = c(8), lambda = 1)
digits.regression(train, test, digits = c(9), lambda = 1)

digits.regression(train, test, transform = transform.phi, digits = c(0), lambda = 1)
digits.regression(train, test, transform = transform.phi, digits = c(1), lambda = 1)
digits.regression(train, test, transform = transform.phi, digits = c(2), lambda = 1)
digits.regression(train, test, transform = transform.phi, digits = c(3), lambda = 1)
digits.regression(train, test, transform = transform.phi, digits = c(4), lambda = 1)
digits.regression(train, test, transform = transform.phi, digits = c(5), lambda = 1)
digits.regression(train, test, transform = transform.phi, digits = c(6), lambda = 1)
digits.regression(train, test, transform = transform.phi, digits = c(7), lambda = 1)
digits.regression(train, test, transform = transform.phi, digits = c(8), lambda = 1)
digits.regression(train, test, transform = transform.phi, digits = c(9), lambda = 1)

## Problem 10
digits.regression(train, test, transform = transform.phi, digits = c(1, 5), lambda = 0.01)
digits.regression(train, test, transform = transform.phi, digits = c(1, 5), lambda = 1)

## Some example calls to digits.regression using the plotting feature
 digits.regression(train, test, digits = c(1, 5), lambda = 1, plotBoundary = TRUE)
 digits.regression(train, test, transform = transform.phi, digits = c(1, 5), lambda = 1, plotBoundary = TRUE)
 digits.regression(train, test, transform = transform.phi, digits = c(1, 5), lambda = 0.01, plotBoundary = TRUE)
