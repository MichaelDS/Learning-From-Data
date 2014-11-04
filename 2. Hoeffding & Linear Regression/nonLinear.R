############### NOISY & NON-LINEARLY SEPARABLE DATA ###############
#
# Noisy targets are generated from the target distribution P(y|x). Each input vector and output pair, (x,y), is 
# generated independently by the joint distribution P(x)*P(y|x).  
#
# A noisy target, y, is defined as the deterministic target f(x) = E[y|x] plus noise y - f(x).  
# A purely deterministic target is a special case of a noisy target: P(y|x) is zero except for y = f(x).
#
############### SIMULATION ###############
#
# This simulation implements linear regression and uses it for classification in the case of noisy and extremely 
# non-linearly separable data. A target function f and a dataset D in d = 2 are created.  X = [-1, 1] x [-1, 1] with 
# uniform probability of picking each x in X.  The deterministic target function for this simulation is set to 
# the following:
#
# f(x) = sign(x1^2 + x2^2 - 0.6)
#
# Noise is simulated by selecting 10% of the data points at random and flipping their response value.
# The inputs, x_n, of the data set are chosen as random points (uniformly in X), and the target function 
# is evaluated on each x_n to get the corresponding output, y_n.  
#
# Attempting to carry out linear regression on the un-transformed data, with feature vector (1, x1, x2), results
# in a poor final hypothesis that pretty much amounts to guessing.  Transforming the data into a new space using
# the feature vector (1, x1, x2, x1*x2, x1^2, x2^2) greatly improves its separability and results in the 
# discovery of a much better final hypothesis via linear regression.


## Function for generating the data and, if specified, the target function y = sign(x1^2 + x2^2 - 0.6)
data.generate <- function(n = 10, ext = 1, generateTarget = FALSE){ 
  # Generate the points
  x1 <- runif(n, -ext, ext)
  x2 <- runif(n, -ext, ext)
  if (!generateTarget)
    return(data.frame(x1, x2))
  
  # Set up a factor for point classification
  y <- sign(x1^2 + x2^2 - 0.6)
  
  # Return the values in a list
  data.frame(x1,x2,y)
  #return(list(data = data,slope = slope, intercept = intercept))
}


## Uses regression to approximate target functions on simulated noisy and non-linearly separable data
## If specified, the data can transformed in order to improve linear separability
## Returns a list containing the average of the in-sample and out-of-sample error measures of the final hypotheses obtained across numTrials as well as a vector of the average of the weights obtained  across numTrials
## Plots the training data used for the last trial and color codes the data points by their responses
nonLinear.simulate <- function(N_train = 1000, N_test = 1000, numTrials = 1000, transform = FALSE) {
  
  # initializing vectors to hold in-sample and out-of-sample error measures
  e_in <- numeric(numTrials)
  e_out <- numeric(numTrials)
  ifelse(transform, g <- numeric(6), g <- numeric(3))  # initializing weight aggregator according to the length of the feature vector
  
  for (i in 1:numTrials) {
    sim <- data.generate(n = N_train, generateTarget = TRUE)
    if(transform) {         # construct an appropriate feature vector depending on whether transformed or un-transformed data is to be used
      X <- matrix(c(rep(1, N_train), sim$x1, sim$x2, sim$x1*sim$x2, sim$x1^2, sim$x2^2), ncol = 6, dimnames = list(c(), c('x0','x1', 'x2', 'x1*x2', 'x1^2', 'x2^2')))
    }
    else {
      X <- matrix(c(rep(1, N_train), sim$x1, sim$x2), ncol = 3, dimnames = list(c(), c('x0','x1', 'x2')))
    }
    y <- sim$y
    noisy <- sample(1:length(y), length(y) * 0.1) # randomly select 10% of the response to be noisy data
    y[noisy] <- y[noisy] * -1        # simulate noise by flipping response at selected indices
    w <- solve(t(X)%*%X)%*%t(X)%*%y  # calculate weights using OLS regression
    g <- g + w                       # aggregate this iteration's weights for averaging later
    y_model <- sign(as.vector(X%*%w))     # apply the final hypothesis to the data
    e_in[i] <- sum(y != y_model)/N_train  # Calculate and store the in-sample error measure
    
    # Test the final hypothesis on independent data
    test <- data.generate(n = N_test, generateTarget = TRUE)
    if(transform) {
      X <- matrix(c(rep(1, N_test), test$x1, test$x2, test$x1*test$x2, test$x1^2, test$x2^2), ncol = 6, dimnames = list(c(), c('x0','x1', 'x2', 'x1*x2', 'x1^2', 'x2^2')))
    }
    else {
      X <- matrix(c(rep(1, N_test), test$x1, test$x2), ncol = 3, dimnames = list(c(), c('x0','x1', 'x2')))
    }
    y2 <- test$y
    noisy <- sample(1:length(y2), length(y2) * 0.1) # randomly select 10% of the response to be noisy data
    y2[noisy] <- y2[noisy] * -1  # simulate noise by flipping response at selected indices
    y_model <- sign(as.vector(X%*%w))
    e_out[i] <- sum(y2 != y_model)/N_test  # calculate and store the out-of-sample error measure
  }
  
  library(ggplot2)
  
  #target <- function(x1) {sqrt(0.6 - x1^2)}
  #target2 <- function(x1) {-sqrt(0.6 - x1^2)}
  plot1 <- qplot(sim$x1, sim$x2, col = as.factor(y), data = data.frame(sim$x1, sim$x2, y), xlab = 'x1', ylab = 'x2', main = 'Noisy and Non-Linearly Separable') 
  #  + geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col=3) 
  
  print(plot1)
  
  list(e_in = mean(e_in), e_out = mean(e_out), w = g/numTrials) # return the averages of the error measures and the weights
}

set.seed(10111)
nonLinear.simulate()  # Problem 8

nonLinear.simulate(transform = TRUE)  # Problems 9 & 10
