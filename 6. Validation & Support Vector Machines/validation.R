############### VALIDATION ###############


## Performs the non-linear transformation phi(x1, x2) = (1, x1, x2, x1^2, x2^2, x1*x2, |x1 - x2|, |x1 + x2|)
## Returns a specified expansion on phi(x1, x2)
## D - Training set with two-dimensional input
## d - desired dimension for the transformed data; determines how far the expand the transformation
transform.phi <- function(D, d = 7) {
  if(d < 3 || d > 7)
    stop('Provided invalid dimension.  Must provide d in the interval [3, 7]')
  x1 <- D[1]
  x2 <- D[2]
  phi <- cbind(1, x1, x2, x1^2, x2^2, x1*x2, abs(x1 - x2), abs(x1 + x2))
  phi[1:(d+1)]
}

## Uses linear regression on a reduced training set to learn a linear classifier, g-, and tests against a validation set
## Also tests g- on a seperate test set for comparison with results from validation set
## Can use non-linear transformations
## Returns measures of the training, validation, and test errors achieved
## D - Training set with column names
## D_test - Test set with column names
## K - Size of the validation set; number of examples from D to be used for validation
## offset - starting index in D from which to extract the validation set 
## transform - A function which returns a non-linear transformation of specified dimension on the input of the training set
## d - desired dimension for the transformed data; determines how far the expand the transformation
## plotBoundary - When set to TRUE, will plot the training and tests sets against the decision boundary; setting this to TRUE will increase the running time of the experiment
regression.classify <- function(D, D_test, K, offset = 0, transform = NULL, d = NULL, plotBoundary = FALSE) {
  y_full <- D[[length(D)]]
  y_train <- y_full[-(offset:(offset+K-1))]
  y_val <- y_full[offset:(offset+K-1)]
  y_test <- D_test[[length(D_test)]]
  if(is.null(transform)) {               # if no transformation is specified, simply use the provided input
    X_full <- as.matrix(cbind(1, D[1:length(D)-1]))
    X_test <- as.matrix(cbind(1, D_test[1:length(D_test)-1]))
  }
  else {                                 # otherwise, perform the non-linear transformation
    X_full <- as.matrix(transform(D, d))
    X_test <- as.matrix(transform(D_test, d))
  }
  
  X_train <- X_full[-(offset:(offset+K-1)), ]
  X_val <- X_full[offset:(offset+K-1), ]
  
  # Compute a final hypothesis using the reduced training set for use with the validation set
  w_ <- solve(t(X_train)%*%X_train)%*%t(X_train)%*%y_train   
  
  # Compute a final hypothesis using the full training set for use with the test set
  w <- solve(t(X_full)%*%X_full)%*%t(X_full)%*%y_full
  
  # Apply the final hypothesis to the inputs and calculate E_in, E_val, and E_out
  y_trainFit <- sign(t(w_)%*%t(X_train))
  E_in <- sum(y_trainFit != y_train)/length(y_trainFit)
  
  y_valFit <- sign(t(w_)%*%t(X_val))
  E_val <- sum(y_valFit != y_val)/length(y_valFit)
  
  y_testFit <- sign(t(w_)%*%t(X_test))
  E_out <- sum(y_testFit != y_test)/length(y_testFit)
  
  if(plotBoundary){
    library(ggplot2)
    library(gridExtra)
    titlePiece <- as.character(d)                                                     # store value of d for plot title
    grid.fit <- expand.grid(list(x1 = seq(-1.5, 1.5, .01), x2 = seq(-1.5, 1.5, .01))) # create a data frame of all combinations of specified x1 and x2 values
    grid.transformed <- transform(grid.fit, d)                                        # transform the grid into the feature space defined by the non-linear transformation
    y <- sign(t(w_)%*%t(grid.transformed))                                            # apply the final hypothesis to every point on the transformed grid in order to obtain predicted y values 
    grid.fit$y <- t(y)                                                                # append the predictions to the un-transformed grid
    
    ## Set up the basic plot and the decision boundary
    base <- ggplot(data = grid.fit, aes(x1, x2, fill = as.factor(y))) + geom_tile() +
      xlab("x1") + ylab("x2") + 
      scale_fill_discrete(limits = c(-1, 1)) +                                # was originally using scale_fill_gradient; discrete makes more sense here
      scale_fill_manual(values = c('gray97', 'lightgoldenrod')) +
      labs(fill = 'Decision Boundary')
    
    ## Plot the training set against the base plot
    plot1 <- base + ggtitle(paste('Training Data: d =', titlePiece)) +
      geom_point(data = D[-(offset:(offset+K-1)), ], aes(x1, x2, colour = as.factor(y))) +
      labs(colour = 'y') + 
      scale_colour_manual(values = c('red', 'blue')) +
      scale_x_continuous(expand = c(0,0)) +
      scale_y_continuous(expand = c(0,0))
    
    ## Plot the validation set against the base plot
    plot2 <- base + ggtitle(paste('Validation Data: d =', titlePiece)) +
      geom_point(data = D[offset:(offset+K-1), ], aes(x1, x2, colour = as.factor(y))) +
      labs(colour = 'y') + 
      scale_colour_manual(values = c('red', 'blue')) +
      scale_x_continuous(expand = c(0,0)) +
      scale_y_continuous(expand = c(0,0))
    
    ## Plot the test set against the base plot
    plot3 <- base + ggtitle(paste('Test Data: d =', titlePiece)) +
      geom_point(data = D_test, aes(x1, x2, colour = as.factor(y))) +
      labs(colour = 'y') + 
      scale_colour_manual(values = c('red', 'blue')) +
      scale_x_continuous(expand = c(0,0)) +
      scale_y_continuous(expand = c(0,0))
    
    # Print the two plots and add a main title
    grid.arrange(plot1, plot2, plot3, ncol = 2, nrow = 2, main = textGrob("Regression with Non-Linear Transformation", gp=gpar(cex=1.5), vjust = 0.7))
  }
  
  list(E_in = E_in, E_val = E_val, E_out = E_out) # return E_in, E_val, and E_out
}

## Read data sets for problems 1-5
train <- read.table('in.txt', col.names = c('x1', 'x2', 'y'))
test <- read.table('out.txt', col.names = c('x1', 'x2', 'y'))

## Problems 1, 2, and 5
regression.classify(train, test, 10, 26, transform.phi, 3)
regression.classify(train, test, 10, 26, transform.phi, 4)
regression.classify(train, test, 10, 26, transform.phi, 5)
regression.classify(train, test, 10, 26, transform.phi, 6)
regression.classify(train, test, 10, 26, transform.phi, 7)

## Problems 3, 4, and 5
regression.classify(train, test, 25, 1, transform.phi, 3)
regression.classify(train, test, 25, 1, transform.phi, 4)
regression.classify(train, test, 25, 1, transform.phi, 5)
regression.classify(train, test, 25, 1, transform.phi, 6)
regression.classify(train, test, 25, 1, transform.phi, 7)

## Sample calls to run experiment with the plotting feature activated
# regression.classify(train, test, 10, 26, transform.phi, 3, plotBoundary = TRUE)
# regression.classify(train, test, 10, 26, transform.phi, 4, plotBoundary = TRUE)
# regression.classify(train, test, 10, 26, transform.phi, 5, plotBoundary = TRUE)
# regression.classify(train, test, 10, 26, transform.phi, 6, plotBoundary = TRUE)
# regression.classify(train, test, 10, 26, transform.phi, 7, plotBoundary = TRUE)
# 
# regression.classify(train, test, 25, 1, transform.phi, 3, plotBoundary = TRUE)
# regression.classify(train, test, 25, 1, transform.phi, 4, plotBoundary = TRUE)
# regression.classify(train, test, 25, 1, transform.phi, 5, plotBoundary = TRUE)
# regression.classify(train, test, 25, 1, transform.phi, 6, plotBoundary = TRUE)
# regression.classify(train, test, 25, 1, transform.phi, 7, plotBoundary = TRUE)
