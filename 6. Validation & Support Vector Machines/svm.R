############### SUPPORT VECTOR MACHINES ###############
#
# Support vector machines (SVM) are some of the most successful learning algorithms to date.  In their canonical form, 
# SVM models are non-probabilistic linear classifiers mapped so that the training examples of the separate categories 
# are divided by a clear margin that is as wide as possible; requiring this margin significantly reduces model
# complexity and optimizes its performance, allowing for the use of sophisticated hypotheses without fully paying the 
# price for them.  New examples are them mapped into that same space and predicted to belong to a category based on 
# which side of the gap they fall on.  In addition to performing linear classification, SVMs can efficiently perform a 
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
#   The problem can is now rewritten as finding the distance between x_n and the plane t(w)%*%x + b = 0, 
#   where |t(w)%*%x_n| + b = 1
#   The vector w is perpendicular to the plane in X space:
#   Take x' and x'' on the plane
#   t(w)%*%x' + b = 0 and t(w)%*%x'' + b = 0  =>  t(w)%*%(x'' - x') = 0
#   The distance between x_n and the plane can be shown to be 1/||w||; take any point x on the plane:
#     The projection of (x_n - x) on w:
#       w^ = w/||w||  =>  distance = |t(w^)%*%(x_n - x)|
#     distance = (1/||w||)%*%(t(w)%*%x_n - t(w)%*%x) 
#              = (1/||w||)%*%(|t(w)%*%x_n + b - t(w)%*%x - b|) = 1/||w||
#   This yields the following optimization problem:
#     Maximize 1/||w|| subject to min(n=1,2,...,N) |t(w)%*%x_n| + b = 1, which is equivalent to
#     Minimize (1/2)*t(w)%*%w subject to y_n*(t(w)%*%x_n + b) >= 1, for n = 1,2, ..., N, w in R^d, b in R
#
# This is a constrained optimization problem; much like regularization optimizes E_in and constrains t(w)%*%w, 
# here t(w)%*%w is optimized and E_in constrained.  
# The Lagrange formulation of the problem is as follows:
#   Minimize L(w, b, alpha) = (1/2)*t(w)%*%w - SIGMA(n - 1, N) alpha_n*(y_n*(t(w)%*%x_n + b) - 1)
#   w.r.t. w and b and maximize w.r.t. each alpha_n >= 0:
#     gradient_w(L) = w - SIGMA(n = 1, N) alpha_n*y_n*x_n = 0
#     dL/db = -SIGMA(n = 1, N) alpha_n*y_n = 0
#     so, w = SIGMA(n = 1, N) alpha_n*y_n*x_n
#     and SIGMA(n = 1, N) alpha_n*y_n = 0
#   Substituting these values simplifies the Lagrangian into following optimization problem;
#   L(alpha) = SIGMA(n = 1, N) alpha_n - (1/2)*SIGMA(n = 1, N) SIGMA(m = 1, N) y_n*y_m*alpha_n*alpha_m*t(x_n)%*%x_m
#   maximize w.r.t. alpha_n >= 0 for n = 1, ..., N and SIGMA(n = 1, N) alpha_n*y_n = 0
#
#   This reduces to the following quadratic programming problem;
#   minimize (1/2)*t(alpha)%*%Q%*%alpha + t(L)%*%alpha
#   subject to t(y)%*%alpha = 0
#   where 0 <= alpha <= infinity, L = t(-1), and
#   Q = cbind(c(y1*y1*t(x1)%*%x1, ..., yN*y1*t(xN)%*%x1), c(y1*y2*t(x1)%*%x2, ...,  yN*y2*t(xN)%*%x2), ....., c(y1*yN*t(x1)%*%xN, ...,  yN*yN*t(xN)%*%xN))
#
# Quadtratic Programming outputs the solution, alpha = alpha_1, alpha_2, ..., alpha_n
# w = SIGMA(n = 1, N) alpha_n*y_n*x_n, (recall that w does not include a bias weight, w_0 in this case)
# KKT condition: For n = 1, 2, ..., N
#   alpha_n*(y_n*(t(w)%*%x_n + b) -1) = 0
# Thus, an alpha_n > 0 indicates that x_n is a support vector (a point that falls on one of the margins of the separating hyperplane defined by w and b; (one of) the nearest point(s) to the plane t(w)%*%x + b = 0)
# because the closest x_n's to the plane achieve the margin y_n*(t(w)%*%x_n + b) = 1
# so w = SIGMA(x is SV) alpha_n*y_n*x_n
# Solve for b using any SV; y_n*(t(w)%*%x_n + b) = 1
#
# Classify points using y_fit = sign(w%*%t(X_test) + b) 
#
# A non-linear transform  can be applied such that X -> Z and the Lagrange formulation is remapped as
# L(alpha) = SIGMA(n = 1, N) alpha_n - (1/2)*SIGMA(n = 1, N) SIGMA(m = 1, N) y_n*y_m*alpha_n*alpha_m*t(z_n)%*%z_m
# The solution is then found is the same manner as before.  The resulting support vectors exist in Z space where
# their margin is maintained.  X contains pre-images of the support vectors.
#
# The generalization behavior of SVMs is described by the following bount:
# E[E_out] <= E[# of SV's]/(N - 1) -> Normal form dividing the the complexity (number of parameters) by approximately
#                                     the number of examples
# It depends on the number of support vectors instead of the dimension of the feature-space used; thus, a complex
# transformation may be used at the cost of a simple model.  In other words, SVMs are able to generate a complex
# hypothesis, h, while maintaining a simple hypothesis set, H, by maximizing the margin; thus, the model's 
# generalization is not as penalized by complex hypothesis sets suggested by high-dimensional feature spaces.
#
############### EXPERIMENT ###############
#
# This simulation implements support vector machines with a hard margin for binary classification and compares their 
# performance against the perceptron learning algorithm (PLA).
#
# A target function f and a dataset D in d = 2 are created.  X = [-1, 1] x [-1, 1] with uniform probability of picking 
# each x in X.  Each run chooses a random line in the plane as the target function by taking the line passing through 
# two random, uniformly distributed points in [-1, 1] x [-1, 1], where one side of the line maps to +1 and the other 
# maps to -1. The inputs, x_n, of the data set are chosen as random points (uniformly in X), and the target function is 
# evaluated on each x_n to get the corresponding output, y_n.  If all of the data points are on one side of the line,
# the data set is discarded and a new one is generated; this is to prevent the SVM from having an infinitely wide 
# margin, which occurs when all of the data points are of the same class.
#
# PLA is implemented almost identically to its implementation in perceptron.R; see that file for more thorough notes 
# on PLA.  On each run, PLA and SVM are trained on the same data set and then tested on a separate data set in order
# to assess and compare their out-of-sample errors.  The number of support vectors found is also returned
# by each run of SVM.  Classification plots for the training and test runs can be constructed for PLA and SVM if
# specified using the plotApproximation parameter.
#
# A comparison of the out-of-sample errors achieved by PLA and SVM across 1000 trials using a training set with 
# 10 data points and a test set with 10000 data points revealed that SVM out-performed PLA in approximately 61 percent
# of the trials.  A similar comparison using 100 training examples in each run resulted in SVM out-performing PLA
# in approximately 63.5 percent of the trials.  
#
# The average number of support vectors used by SVM across 1000 trials with training and test sets of size 10 and
# 1000, respectively, was found to be approximately 3.

############### IMPLEMENTATION ###############

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
SVM <- function(train, test, plotApproximation = FALSE){
  library(LowRankQP)
  
  y <- train$data$y
  X <- as.matrix(train$data[-(ncol(train$data))]) # extract the input vectors; not cbinding an intercept term, x0, in order to simplify calculations
  L <- rep(-1, nrow(X))                           # linear term; vector appearing in the quadratic function to be minimized
  Q <- (y*X)%*%t(y*X)                             # quadratic term; matrix appearing in quadratic function to be minimized; vectorized construction is more efficient than loops
  constraintMatrix <- t(y)                        # matrix defining the constraints under which to minimize the quadratic function
  constraintValues <- 0                           # vector of constraint values under which to minimize the quadratic function
  upperBounds <- rep(10000, nrow(X))              # vector of upper bounds on the Lagrange multipliers; 0 <= alpha_i <= upperLimit; theoretical upper limit is +infitity, using 10000 instead
  solution <- LowRankQP(Vmat = Q, dvec = L, Amat = constraintMatrix, bvec = constraintValues, uvec = upperBounds, method = 'LU')  # find solution that minimizes the quadratic function
  alpha <- zapsmall(solution$alpha, digits = 6)   # extract the Lagrange multiplier; values greater than 0 correspont to support vectors; zapsmall() rounds extremely small alphas down to zero
  sv_indices <- which(alpha != 0)                 # compute the indices of the support vectors
  w <- colSums(alpha[sv_indices] * y[sv_indices] * X[sv_indices, ]) # use alpha to compute w (which does not include w0, the bias term, in this case)
  
  #     w <- numeric(ncol(X))                                       # simpler with colSums
  #     for(n in sv_indices) {                         
  #       w <- w + alpha[n]*y[n]*X[n, ]
  #     }
  
  b = as.numeric(1/y[sv_indices[1]] - t(w)%*%X[sv_indices[1], ])    # solve for the bias term; can be done using any of the support vectors
  
  X_test <- as.matrix(test[, -ncol(test)])
  y_test <- test$y                 
  y_fit <- sign(w%*%t(X_test) + b)                # classify points according to the final hypothesis
  numSV <- length(sv_indices)                     # store the number of support vectors found
  
  ## TESTING PURPOSES
  #   library(e1071)
  #   t <- svm(y~x1+x2, data = train$data, scale = FALSE, kernel = 'linear', cost = 1000, type = 'C-classification')
  #   y_fit <- predict(t, newdata = X_test)
  #   print(t$SV)
  #   print(X[sv_indices, ])
  
  E_out <- sum(y_test != y_fit)/nrow(X_test)       # store the misclassification error 
  
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
    abline(train$intercept, train$slope, col = 'green3')
    abline(-b/w[2], -w[1]/w[2])
    abline(-(b+1)/w[2], -w[1]/w[2], lty = 2)
    abline(-(b-1)/w[2], -w[1]/w[2], lty = 2)
  }
  
  # return the out-of-sample error and the number of support vectors
  list(E_out = E_out, num_support_vectors = numSV, w = w, b = b) 
}

SVM.simulate <- function(N_train = 10, N_test = 1000, numTrials = 1000, simulation = data.generate, plotApproximation = FALSE) {
  E_out <- numeric(numTrials)
  num_SV <- numeric(numTrials)
  for(i in 1:numTrials) {
    train <- simulation(N_train, generateTarget = TRUE)
    test <- simulation(N_test)
    test$y <- as.numeric(test[, 'x1'] * train$slope + train$intercept > test[, 'x2']) * 2 - 1
    result <- SVM(train, test, plotApproximation)
    E_out[i] <- result$E_out
    num_SV[i] <- result$num_support_vectors
  }
  list(E_out = mean(E_out), avg_num_support_vectors = mean(num_SV))
}

## Uses the perceptron(pocket) learning algorithm to approximate target functions on simulated data
## Returns a list containing the number of iterations taken by PLA and the out-of-sample error averaged across numTrials
## Plots the target function and final hypothesis of the last trial against training data and against test data
PLA <- function(train, test, maxIterations = Inf, plotApproximation = FALSE) {
  
  input <- as.matrix(cbind(1, train$data[c(1,2)]))            # create the input matrix
  
  w <- c(0,0,0)                                               # initialize the weight vector
  
  res <- as.vector(input%*%w)                                 # apply the initial weights to the input to get the initial hypothesis
  
  best <- list(sum(sign(res) != train$data$y)/length(res), w) # initialize list to keep track of the best in-sample error achieved so far and the weight vector that produced it
  
  k <- 0                                                      # initializing iteration counter
  
  while (any(sign(res) != train$data$y) && k < maxIterations) # as long as any of the elements of res do not match the true output, y, and the iterations threshold has not been reached
    # the PLA algorithm continues to iterate  
  {                                            
    misclassified <- which(sign(res) != train$data$y)         # get the indices of the points for which hypothesis is wrong
    ifelse (length(misclassified) == 1, n <- misclassified, n <- sample(misclassified,1))  # randomly choose one of these points
    w <- w + train$data$y[n]*input[n, ]                       # update the weights
    res <- apply(input, 1, function(x) t(w)%*%x)              # use new weights to update the hypothesis function
    e_in <- sum(sign(res) != train$data$y)/length(res)        # calculate in-sample error
    if (e_in < best[[1]]) {
      best <- list(e_in, w)                                   # if a the current weight vector is better than the previous best, store it
    }
    k <- k + 1                                                # increment iteration count
  }
  
  w <- best[[2]]                                              # selecting the best weight vector discovered by the algorithm
  
  iterations <- k                                             # store the number of iterations needed in this run
  
  X_test <- cbind(1, as.matrix(test[, -ncol(test)]))
  y_test <- test$y
  y_fit <- sign(w%*%t(X_test))                                # classify points according to the final hypothesis
  
  #  g <- as.numeric(test$x1 * (-w[2]/w[3]) - w[1]/w[3] > test$x2) * 2 - 1 # this method flips the classifications on occasional iterations for some reason
  
  ##  TESTING PURPOSES
  #   print(sum(y_fit == g))
  #   if(sum(y_fit == g) == 0) {
  #     print(sum(y_test != g)/N_test)
  #     print('---------')
  #     print(sum(y_test != y_fit)/N_test)
  #     library(ggplot2)
  #     library(gridExtra)
  #     plot1 <- qplot(x1,x2,col= as.factor(y), data = train$data) + geom_abline(intercept = train$intercept, slope = train$slope) +
  #       geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col=3)
  #     plot2 <- qplot(x1,x2,col= as.factor(y_fit), data = test) + geom_abline(data = train$data, intercept = train$intercept, slope = train$slope) +
  #       geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col=3)
  #     plot3 <- qplot(x1,x2,col= as.factor(g), data = test) + geom_abline(data = train$data, intercept = train$intercept, slope = train$slope) +
  #       geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col=3)
  #     grid.arrange(plot1, plot2, plot3, ncol = 3)
  #   }
  
  E_out <- sum(y_test != y_fit)/nrow(X_test)                  # store the misclassification error from this run
  
  if(plotApproximation) {
    # Plot the points and f and g functions from the last iteration (purely illustrative purposes)
    library(ggplot2)
    library(gridExtra)
    plot1 <- qplot(x1,x2,col= as.factor(y), data = train$data, main = 'Perceptron - Training') + geom_abline(intercept = train$intercept, slope = train$slope) +
      geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col='yellow')
    plot2 <- qplot(x1,x2,col= as.factor(y), data = test, main = 'Perceptron - Test') + geom_abline(data = train$data, intercept = train$intercept, slope = train$slope) +
      geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col='yellow')
    grid.arrange(plot1, plot2, ncol = 2)
  }
  
  # return the out-of-sample error and the number of iterations taken by PLA
  list(E_out = E_out, num_iterations = iterations) 
}

PLA.simulate <- function(N_train = 10, N_test = 1000, numTrials = 1000, maxIterations = Inf, simulation = data.generate, plotApproximation = FALSE) {
  E_out <- numeric(numTrials)
  num_iter <- numeric(numTrials)
  for(i in 1:numTrials) {
    train <- simulation(N_train, generateTarget = TRUE)
    test <- simulation(N_test)
    test$y <- as.numeric(test[, 'x1'] * train$slope + train$intercept > test[, 'x2']) * 2 - 1
    result <- PLA(train, test, maxIterations, plotApproximation)
    E_out[i] <- result$E_out
    num_iter[i] <- result$num_iterations
  }
  list(E_out = mean(E_out), avg_num_iterations = mean(num_iter))
}

SVM_PLA.compare <- function(N_train = 10, N_test = 10000, numTrials = 1000, maxIterations = Inf, simulation = data.generate, plotApproximation = FALSE) {
  count <- 0
  for(i in 1:numTrials) {
    train <- simulation(N_train, generateTarget = TRUE)
    test <- simulation(N_test)
    test$y <- as.numeric(test[, 'x1'] * train$slope + train$intercept > test[, 'x2']) * 2 - 1
    SVM_error <- SVM(train, test, plotApproximation)$E_out
    PLA_error <- PLA(train, test, maxIterations, plotApproximation)$E_out
    if(SVM_error < PLA_error)
      count <- count + 1
  }
  count/numTrials
}

set.seed(10111)

SVM_PLA.compare()              # Problem 8
SVM_PLA.compare(N_train = 100) # Problem 9
SVM.simulate(N_train = 100)    # Problem 10

## Sample calls to run SVM with the plotting feature activated
# SVM.simulate(N_train = 10, numTrials = 1, plotApproximation = TRUE)    
# SVM.simulate(N_train = 100, numTrials = 1, plotApproximation = TRUE)    

## Sample calls to run the PLA simulation with the plotting feature activated
# PLA.simulate(N_train = 10, numTrials = 1, plotApproximation = TRUE)
# PLA.simulate(N_train = 100, numTrials = 1, plotApproximation = TRUE)
