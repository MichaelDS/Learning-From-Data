# Supervised learning via linear regression is designed for use with real-valued output, however, it can also provide 
# good results when used for classification.  An advantage of linear regression is the simplicity of the algorithm for 
# computing the final hypothesis.  This "one-step learning" is carried out as follows:
#
# Given the N x n data matrix, X, and the N x 1 output matrix, y, compute the weight vector
#
# w = pseudoInverse(X) x y = (t(X) x X)^-1 x t(X) x y
#
# This weight vector minimizes the error measure (sum of squares), thus, the final hypothesis is computed.   
# This weight vector can also be used as an initial weight vector for a learning algorithm designed for 
# classification, such as PLA.  This provides the benefit of the more precise final hypothesis provided by PLA
# along with a lower computation time than would have been required to run the PLA algorithm alone; this is 
# because regression provides PLA with a good initial approximation, cheaply taking care of much of the searching 
# that the PLA algorithm would otherwise have done.
#
# This simulation implements and examines linear regression for classification.  A target function f and a dataset D 
# in d = 2 are created.  X = [-1, 1] x [-1, 1] with uniform probability of picking each x in X.  Each run chooses a 
# random line in the plane as the target function by taking the line passing through two random, uniformly distributed 
# points in [-1, 1] x [-1, 1], where one side of the line maps to +1 and the other maps to -1. The inputs, x_n, of the 
# data set are chosen as random points (uniformly in X), and the target function is evaluated on each x_n to get the 
# corresponding output, y_n. Regression is then be used to learn from this data and the final hypothesis is tested
# against an independently generated data set.  This can be done  with or without any additional adjustment to the 
# final hypothesis by PLA.


## Function for generating the data and, if specified, a target function
data.generate <- function(n = 10, ext = 1, generateTarget = FALSE){ 
  # Generate the points
  x1 <- runif(n, -ext, ext)
  x2 <- runif(n, -ext, ext)
  if (!generateTarget)
    return(data.frame(x1, x2))
  
  # Draw a random line in the area (target function)
  point <- runif(2, -ext, ext)
  point2 <- runif(2, -ext, ext)
  slope <- (point2[2] - point[2]) / (point2[1] - point[1])
  intercept <- point[2] - slope * point[1]
  
  # Set up a factor for point classification
  y <- as.numeric(x1 * slope + intercept > x2) * 2 - 1
  
  # Return the values in a list
  data <- data.frame(x1,x2,y)
  return(list(data = data,slope = slope, intercept = intercept))
}


## Uses regression to approximate target functions on simulated data
## If specified, the resulting regression model is used as a starting point for the perceptron(pocket) learning algorithm in order to further improve the approximation
## Returns a list containing the average of the in-sample and out-of-sample error measures of the final hypotheses obtained across numTrials
## Plots the target function and final hypothesis of the last trial against training data and against test data
classificationByRegression.simulate <- function(N_train = 100, N_test = 1000, numTrials = 1000, PLA = FALSE, maxIterations = Inf) {
  
  # initializing vectors to hold in-sample and out-of-sample error measures and the number of iterations taken by PLA, if applicable
  e_in <- numeric(numTrials)  
  e_out <- numeric(numTrials) 
  iterations <- as.numeric(rep(NA, numTrials)) 
  
  for (i in 1:numTrials) {
    sim <- data.generate(N_train, generateTarget = TRUE)
    X <- matrix(c(rep(1, N_train), sim$data$x1, sim$data$x2), ncol = 3, dimnames = list(c(), c('x0','x1', 'x2')))
    y <- sim$data$y
    w <- solve(t(X)%*%X)%*%t(X)%*%y        # calculating weights that minimize E_in (squared error)
    y_model <- sign(as.vector(X%*%w))      # apply the hypothesis function to the data
    e_train <- sum(y != y_model)/N_train   # calculate in-sample error
    
    if(PLA) {
      best <- list(e_train, w)  # initializing list to keep track of the best in-sample error achieved so far and the weight vector that produced it
      k <- 0                    # initializing iteration counter
      while (any(sign(y_model) != y) && k < maxIterations) # as long as any of the elements of y_model do not match the true output, y, and the iterations threshold has not been reached
                                                           # the PLA algorithm continues to iterate
      {                                           
        misclassified <- which(sign(y_model) != y)         # getting the indices of the points for which hypothesis is wrong
        ifelse (length(misclassified) == 1, n <- misclassified, n <- sample(misclassified, 1))  # randomly choose one of these points
        w <- w + y[n] * X[n,]                              # update the weights
        y_model <- apply(X, 1, function(x) t(w)%*%x)       # use new weights to update the hypothesis function
        e_train <- sum(sign(y_model) != y)/length(y_model) # calculate in-sample error
        if (e_train < best[[1]]) {
          best <- list(e_train, w)                         # if a the current weight vector is better than the previous best, store it
        }
        k <- k+1  # increment iteration count
      }
      
      e_train <- best[[1]] # updating e_in
      w <- best[[2]]       # selecting the best weight vector discovered by the algorithm
      
      iterations[i] <- k   # store the number of iterations needed in this run
      
    }
    e_in[i] <- e_train     # store the in-sample error
    
    X <- as.matrix(cbind(1, data.generate(N_test)))  # generate test data set to examine out of sample performance
    y <- as.numeric(X[, 'x1'] * sim$slope + sim$intercept > X[, 'x2']) * 2 - 1  # classify the test points using the target function generated with the training data set
    y_model <- sign(as.vector(X%*%w))                # apply the hypothesis function to the test data
    e_out[i] <- sum(y != y_model)/N_test             # calculate and store out-of-sample error
  }
  
  library(ggplot2)
  library(gridExtra)
  
  plot1 <- qplot(sim$data$x1, sim$data$x2, col= as.factor(sim$data$y), data = sim$data, xlab = 'x1', ylab = 'x2', main = 'Training Data') + 
    geom_abline(intercept = sim$intercept, slope = sim$slope) +
    geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col=3)
  
  test <- data.frame(x1 = X[, 'x1'], x2 = X[, 'x2'], y = y)
  plot2 <- qplot(test$x1, test$x2, col= as.factor(test$y), data = test, xlab = 'x1', ylab = 'x2', main = 'The Same Hypothesis and Target Function Against Test Data') + 
    geom_abline(intercept = sim$intercept, slope = sim$slope) +
    geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col=3)
  
  grid.arrange(plot1, plot2, ncol=2)
  
  list(e_in = mean(e_in), e_out = mean(e_out), iterations = mean(iterations)) # return the averages of the error measures and the number of iterations run by the PLA, if applicable
}

set.seed(10111)
classificationByRegression.simulate()  # Problems 5 & 6

classificationByRegression.simulate(N_train = 10, PLA = TRUE)  # Problem 7
