############### SUPPORT VECTOR MACHINES ###############
#
# Support vector machines (SVM) enable learning via complex models at the price of a simple one and are one of the 
# most successful learning algorithms to date.  The canonical SVM model is a non-probabilistic linear classifier 
# mapped so that the training examples of the separate categories are divided by a clear margin that is as wide as
# possible.  New examples are them mapped into that same space and predicted to belong to a category based on which
# side of the gap they fall on.  In addition to performing linear classification, SVMs can efficiently perform a
# non-linear classification by using a kernel method, implicitly mapping inputs into high-dimensional feature spaces.
# Additionally, SVMs have been extended to perform other tasks including multi-class classification and regression.
# 
############### SVM AS A LINEAR CLASSIFIER ###############
#
# SVMs achieve better linear separation that perceptrons by finding the maximum margin classifier, or equivalently,
# the perceptron of optimal stability.  Intuitively, a larger margin can generally be associated with a lower 
# generalization error for the classifier, as wider margins imply fewer dichotomies.
#
# Finding w with a large margin:
#   Suppose x_n = the nearest point to the plane t(w)%*%x = 0
#   Normalize w such that |t(w)%*%x_n| = 1
#   Pull w_0 out of w such that w = (w_1, ..., w_d) apart from b
#   The plane is now t(w)%*%x + b = 0  (no x_0)
#   The distance between x_n and the plane can be shown to be 1/||w||
#   This yields the following optimization problem:
#     Maximize 1/||w|| subject to min(n=1,2,...,N) |t(w)%*%x_n| + b = 1, which is equivalent to
#     Minimize (1/2)*t(w)%*%w subject to y_n*(t(w)%*%x_n + b) >= 1, for n = 1,2, ..., N


## Generates the input data and, if specified, a target function
## Ensures that the response values do not all belong to a single class
data.generate <- function(N = 10, limits = c(-1, 1), generateTarget = FALSE){ 
  repeat{
    # generate the points
    x1 <- runif(N, limits[1], limits[2])
    x2 <- runif(N, limits[1], limits[2])
    if (!generateTarget)
      return(data.frame(x1, x2))
    
    # draw a random line in the area (target function)
    point <- runif(2, limits[1], limits[2])
    point2 <- runif(2, limits[1], limits)
    slope <- (point2[2] - point[2]) / (point2[1] - point[1])
    intercept <- point[2] - slope * point[1]
    
    # set up a factor for point classification
    y <- as.numeric(x1 * slope + intercept > x2) * 2 - 1
    
    if(abs(sum(y)) != length(y))  # make sure the points are not all on one side of the line
      break
  }
  
  data <- data.frame(x1,x2,y) # put the training set together
  
  return(list(data = data, slope = slope, intercept = intercept)) # return the values in a list
}  

## 1/2*t(alpha)%*%Q%*%alpha - t(1)%*%alpha
## w & b
SVM.simulate <- function(N_train = 10, N_test = 1000, numTrials = 1000, simulation = data.generate, plotApproximation = FALSE, trialResults = FALSE){
  library(LowRankQP)
  E_out <- numeric(numTrials)
  numSV <- numeric(numTrials)
  
  for(i in 1:numTrials) {
    sim <- simulation(N = N_train, generateTarget = TRUE) # generate points and target function
    y <- sim$data$y
    X <- as.matrix(sim$data[-(ncol(sim$data))]) # extract the input vectors; not cbinding an intercept term, x0, in order to simplify calculations
    L <- rep(-1, N_train)                       # linear term; vector appearing in the quadratic function to be minimized
    Q <- (y*X)%*%t(y*X)                         # quadratic term; matrix appearing in quadratic function to be minimized; vectorized construction is more efficient than loops
    constraintMatrix <- t(y)                    # matrix defining the constraints under which to minimize the quadratic function
    constraintValues <- 0                       # vector of constraint values under which to minimize the quadratic function
    upperBounds <- rep(10000, N_train)          # vector of upper bounds on the Lagrange multipliers; 0 <= alpha_i <= upperLimit; theoretical upper limit is +infitity, using 10000 instead
    
    solution <- LowRankQP(Vmat = Q, dvec = L, Amat = constraintMatrix, bvec = constraintValues, uvec = upperBounds, method = 'LU')  # find solution that minimizes the quadratic function
    alpha <- zapsmall(solution$alpha, digits = 6)  # extract the Lagrange multiplier; values greater than 0 correspont to support vectors; zapsmall() rounds extremely small alphas down to zero
    sv_indices <- which(alpha != 0)                # compute the indices of the support vectors
    
    w <- colSums(alpha[sv_indices] * y[sv_indices] * X[sv_indices, ]) # use alpha to compute w (which does not include w0, the bias term, in this case)
    
#     w <- numeric(ncol(X))                                           # simpler with colSums
#     for(n in sv_indices) {                         
#       w <- w + alpha[n]*y[n]*X[n, ]
#     }

    b = as.numeric(1/y[sv_indices[1]] - t(w)%*%X[sv_indices[1], ])  # solve for the bias term; can be done using any of the support vectors
    
    X_test <- as.matrix(simulation(N_test))
    y_test <- as.numeric(X_test[, 'x1'] * sim$slope + sim$intercept > X_test[, 'x2']) * 2 - 1  # classify points according to the target function f
    y_fit <- sign(w%*%t(X_test) + b)               # classify points according to the final hypothesis
    numSV <- length(sv_indices)                    # store the number of support vectors from this run
    
    ##testing purposes
    #     library(e1071)
    #     t <- svm(y~x1+x2, data = sim$data, scale = FALSE, kernel = 'linear', cost = 1000, type = 'C-classification')
    #     y_fit <- predict(t, newdata = X_test)
    
    E_out[i] <- sum(y_test != y_fit)/N_test        # store the misclassification error from this run
  }
  
  if(plotApproximation) {                          # construct classification plots for the training and test runs
    par(mfrow = c(1,2))
    gridRange <- apply(X, 2, range)
    x1 <- seq(from = gridRange[1, 1], to = gridRange[2, 1], length = 75)
    x2 <- seq(from = gridRange[1, 2], to = gridRange[2, 2], length = 75)
    xgrid <- expand.grid(x1 = x1, x2 = x2)
    ygrid <- sign(w%*%t(xgrid) + b)
    plot(xgrid, col = c('red', 'blue')[as.numeric(as.factor(ygrid))], pch = 20, cex = 0.2, main = 'SVM Classification Plot - Training')
    points(X, col = y + 3, pch = 19)
    points(X[sv_indices, ], pch = 5, cex = 2)
    #abline(sim$intercept, sim$slope, col = 'green')
    abline(-b/w[2], -w[1]/w[2])
    abline(-(b+1)/w[2], -w[1]/w[2], lty = 2)
    abline(-(b-1)/w[2], -w[1]/w[2], lty = 2)
    
    gridRange <- apply(X_test, 2, range)
    x1 <- seq(from = gridRange[1, 1], to = gridRange[2, 1], length = 75)
    x2 <- seq(from = gridRange[1, 2], to = gridRange[2, 2], length = 75)
    xgrid <- expand.grid(x1 = x1, x2 = x2)
    ygrid <- sign(w%*%t(xgrid) + b)
    plot(xgrid, col = c('red', 'blue')[as.numeric(as.factor(ygrid))], pch = 20, cex = 0.2, main = 'SVM Classification Plot - Test')
    points(X_test, col = y_test + 3, pch = 19, cex = 0.5)
    abline(sim$intercept, sim$slope, col = 'green3')
    abline(-b/w[2], -w[1]/w[2])
    abline(-(b+1)/w[2], -w[1]/w[2], lty = 2)
    abline(-(b-1)/w[2], -w[1]/w[2], lty = 2)
  }
  
  if(trialResults)
    return(E_out) # if specified, return the vector of E_out computed each trial
  
  # return the estimated expected out-of-sample error, average number of support vectors
  list(E_out = mean(E_out), avg_num_support_vectors = mean(numSV)) 
}

## Uses the perceptron(pocket) learning algorithm to approximate target functions on simulated data
## Returns a list containing the number of iterations taken by PLA and the out-of-sample error averaged across numTrials
## Plots the target function and final hypothesis of the last trial against training data and against test data
PLA.simulate <- function(N_train = 10, N_test = 10000, numTrials = 1000, maxIterations = Inf, simulation = data.generate, plotApproximation = FALSE, trialResults = FALSE) {
  iterations <- numeric(numTrials)    # initializing the iteration and misclassification probability vectors
  E_out <- numeric(numTrials)
  
  # number of times to repeat the experiment is specified by numTrials
  
  for (i in 1:numTrials){
    sim <- simulation(N = N_train, generateTarget = TRUE) # generate points and target function
    input <- as.matrix(cbind(1, sim$data[c(1,2)])) # create the input matrix
    
    w <- c(0,0,0)  # initialize the weight vector
    
    res <- as.vector(input %*% w) # apply the initial weights to the input to get the initial hypothesis
    
    best <- list(sum(sign(res) != sim$data$y)/length(res), w)  # initialize list to keep track of the best in-sample error achieved so far and the weight vector that produced it
    
    k <- 0  # initializing iteration counter
    
    while (any(sign(res) != sim$data$y) && k < maxIterations) # as long as any of the elements of res do not match the true output, y, and the iterations threshold has not been reached
      # the PLA algorithm continues to iterate  
    {                                           
      misclassified <- which(sign(res) != sim$data$y)  # get the indices of the points for which hypothesis is wrong
      ifelse (length(misclassified) == 1, n <- misclassified, n <- sample(misclassified,1))  # randomly choose one of these points
      w <- w + sim$data$y[n]*input[n, ]                # update the weights
      res <- apply(input, 1, function(x) t(w)%*%x)           # use new weights to update the hypothesis function
      e_in <- sum(sign(res) != sim$data$y)/length(res) # calculate in-sample error
      if (e_in < best[[1]]) {
        best <- list(e_in, w) # if a the current weight vector is better than the previous best, store it
      }
      k <- k + 1          # increment iteration count
    }
    
    w <- best[[2]]      # selecting the best weight vector discovered by the algorithm
    
    iterations[i] <- k  # store the number of iterations needed in this run
    
    new.data <- simulation(N_test)  # generating the test points in order to examine out-of-sample performance
    f <- as.numeric(new.data$x1 * sim$slope + sim$intercept > new.data$x2) * 2 - 1  # classify points according to the target function f
    g <- as.numeric(new.data$x1 * (-w[2]/w[3]) - w[1]/w[3] > new.data$x2) * 2 - 1   # classify points according to the hypothesized function g, using the 
                                                                                    # final weights provided by PLA            
    
    E_out[i] <- sum(f != g)/N_test  # store the misclassification error from this run
  }
  
  if(plotApproximation) {
    # Plot the points and f and g functions from the last iteration (purely illustrative purposes)
    library(ggplot2)
    plot1 <- qplot(x1,x2,col= as.factor(y), data = generated$data) + geom_abline(intercept = generated$intercept, slope = generated$slope) +
      geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col=3)
    print(plot1)
  }
  
  if(trialResults)
    return(E_out) # if specified, return the vector of E_out computed each trial
  
  # return the estimated expected out-of-sample error, average number of iterations taken by PLA
  list(E_out = mean(E_out), avg_num_iterations = mean(iterations)) 
}

SVM_PLA.compare <- function(N_train = 10, N_test = 1000, numTrials = 1000, maxIterations = Inf, simulation = data.generate, plotApproximation = FALSE, trialResults = TRUE) {
  SVM_error <- SVM.simulate(N_train, N_test, numTrials, simulation, plotApproximation, trialResults)
  PLA_error <- PLA.simulate(N_train, N_test, numTrials, maxIterations, simulation, plotApproximation, trialResults)
  sum(SVM_error < PLA_error)/numTrials
}

set.seed(10111)

SVM_PLA.compare()              # Problem 8
SVM_PLA.compare(N_train = 100) # Problem 9
SVM.simulate(N_train = 100)    # Problem 10

## Sample calls to run the SVM simulation with the plotting feature activated
# SVM.simulate(N_train = 10)    
# SVM.simulate(N_train = 100)    

