############### NON-LINEAR CLASSIFICATION VIA THE KERNEL TRICK ###############

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

transform.phi <- function(D) {
  z1 <- D[['x2']]^2 - 2*D[['x1']] - 1
  z2 <- D[['x1']]^2 - 2*D[['x2']] + 1
  data.frame(z1 = z1, z2 = z2, y = D[['y']])
}

kernel.quadratic <- function(xx) {
  (1 + xx)^2
}

SVM <- function(train, test = NULL, kernel = NULL, C = 10^6, plotApproximation = FALSE){
  library(LowRankQP)
  
  y <- train$y
  X <- as.matrix(train[-(ncol(train))])           # extract the input vectors; not cbinding an intercept term, x0, in order to simplify calculations
  L <- rep(-1, nrow(X))                           # linear term; vector appearing in the quadratic function to be minimized
  if(!is.null(kernel)) {
    Kxx <- X%*%t(X)
    Kxx <- kernel(Kxx)
    Qy <- y%*%t(y)
    Q <- Qy*Kxx
  }
  else
    Q <- (y*X)%*%t(y*X)                           # quadratic term; matrix appearing in quadratic function to be minimized; vectorized construction is more efficient than loops
  
  constraintMatrix <- t(y)                        # matrix defining the constraints under which to minimize the quadratic function
  constraintValues <- 0                           # vector of constraint values under which to minimize the quadratic function
  upperBounds <- rep(C, nrow(X))                  # vector of upper bounds on the Lagrange multipliers; 0 <= alpha_i <= upperLimit; theoretical upper limit is +infitity, use 10^6 instead for hard margin
  solution <- LowRankQP(Vmat = Q, dvec = L, Amat = constraintMatrix, bvec = constraintValues, uvec = upperBounds, method = 'LU')  # find solution that minimizes the quadratic function
  alpha <- zapsmall(solution$alpha, digits = 6)   # extract the Lagrange multiplier; values greater than 0 correspont to support vectors; zapsmall() rounds extremely small alphas down to zero
  sv_indices <- which(alpha != 0)                 # compute the indices of the support vectors
  numSV <- length(sv_indices)                     # store the number of support vectors found
  
  if(is.null(kernel)) {
    w <- colSums(alpha[sv_indices] * y[sv_indices] * X[sv_indices, ])  # use alpha to compute w (which does not include w0, the bias term, in this case)
    b <- as.numeric(as.matrix(1/y[sv_indices[1]] - t(w)%*%X[sv_indices[1], ]))    # solve for the bias term; can be done using any of the support vectors
    g <- function(newX, decision.values = FALSE) {                                # define a function for making predictions based on the final hypothesis
      decision <- w%*%t(newX) + b
      if(decision.values) 
        return(list(response = sign(decision), decisionVals = decision))
      else
        sign(decision)
    }
  }
  else {
    b <- y[sv_indices[1]] - colSums(as.matrix(alpha[sv_indices] * y[sv_indices] * Kxx[sv_indices, sv_indices[1]])) # solve for the bias term in terms of the kernel function; can be done using any of the support vectors
    g <- function(newX, decision.values = FALSE) {                                                                 # define a function for making predictions based on the final hypothesis terms of the kernel function      
      Kxx <- X%*%t(newX)                                 # kernel matrix is constructed using the inner products of the training input and new input
      Kxx <- kernel(Kxx)
      decision <- colSums(as.matrix(alpha[sv_indices] * y[sv_indices] * Kxx[sv_indices, ])) + b
      if(decision.values) 
        return(list(response = sign(decision), decisionVals = decision))
      else
        sign(decision)
    } 
  }
  
  y_fit <- g(X)                                      # classify points according to the final hypothesis
  E_in <- sum(y != y_fit)/nrow(X)                    # store the in-sample misclassification error
  
  if(!is.null(test)) {
    X_test <- as.matrix(test[, -ncol(test)])
    y_test <- test$y                 
    y_fit <- g(X_test)                               # classify points according to the final hypothesis
    E_out <- sum(y_test != y_fit)/nrow(X_test)       # store the misclassification error 
  }
  
  if(plotApproximation) {                            # construct classification plots for the training and test runs
    if(!is.null(test))
      par(mfrow = c(1,2))    
    
    gridRange <- apply(X, 2, range)
    x1 <- seq(from = gridRange[1, 1], to = gridRange[2, 1], length = 75)
    x2 <- seq(from = gridRange[1, 2], to = gridRange[2, 2], length = 75)
    xgrid <- expand.grid(x1 = x1, x2 = x2)
    fit <- g(as.matrix(xgrid), decision.values = TRUE)
    ygrid <- fit$response
    plot(xgrid, col = c('red', 'blue')[as.numeric(as.factor(ygrid))], pch = 20, cex = 0.2, main = 'SVM Classification Plot - Training')
    points(X, col = y + 3, pch = 19)
    points(X[sv_indices, ], pch = 5, cex = 2)
    
    if(is.null(kernel)) {
      #abline(sim$intercept, sim$slope, col = 'green')
      if(w[2] == 0) {
        abline(v = -b/w[1])
        abline(v = -(b+1)/w[1], lty = 2)
        abline(v = -(b-1)/w[1], lty = 2)
      }
      else {
        abline(-b/w[2], -w[1]/w[2])
        abline(-(b+1)/w[2], -w[1]/w[2], lty = 2)
        abline(-(b-1)/w[2], -w[1]/w[2], lty = 2)
      }
    }
    else {
      decisionValues <- fit$decisionVals
      contour(x1, x2, matrix(decisionValues, length(x1), length(x2)), level = 0, lwd = 1.5, drawlabels = FALSE, add=TRUE)
    }

    if(!is.null(test)) {
      gridRange <- apply(X_test, 2, range)
      x1 <- seq(from = gridRange[1, 1], to = gridRange[2, 1], length = 75)
      x2 <- seq(from = gridRange[1, 2], to = gridRange[2, 2], length = 75)
      xgrid <- expand.grid(x1 = x1, x2 = x2)
      fit <- g(as.matrix(xgrid), decision.values = TRUE)
      ygrid <- fit$response
      plot(xgrid, col = c('red', 'blue')[as.numeric(as.factor(ygrid))], pch = 20, cex = 0.2, main = 'SVM Classification Plot - Test')
      points(X_test, col = y_test + 3, pch = 19, cex = 0.5)
      
      if(is.null(kernel)) {
        abline(train$intercept, train$slope, col = 'green3')
        if(w[2] == 0) {
          abline(v = -b/w[1])
          abline(v = -(b+1)/w[1], lty = 2)
          abline(v = -(b-1)/w[1], lty = 2)
        }
        else {
          abline(-b/w[2], -w[1]/w[2])
          abline(-(b+1)/w[2], -w[1]/w[2], lty = 2)
          abline(-(b-1)/w[2], -w[1]/w[2], lty = 2)
        }
      }
      else {
        decisionValues <- fit$decisionVals
        contour(x1, x2, matrix(decisionValues, length(x1), length(x2)), level = 0, lwd = 1.5, drawlabels = FALSE, add=TRUE)
      }
    }  
  }
  # return the error measures and the number of support vectors
  return(list(E_out = ifelse(is.null(test), NA, E_out), E_in = E_in, num_support_vectors = numSV, w = ifelse(is.null(kernel), w, NA), b = b)) 
}

D_train <- data.frame(x1 = c(1,0,0,-1,0,0,-2), x2 = c(0,1,-1,0,2,-2,0), y = c(-1,-1,-1,1,1,1,1))
plot(D_train[-3], col = c('red', 'blue')[as.numeric(as.factor(D_train$y))], pch = 20)

D_transformed <- transform.phi(D_train)

## Problem 11 - It's clear from the plot that the separating hyperplane that maximizes the margin in the Z space can
## be specified as 0*z2 + 1*z1 + 0.5 = 0
plot(D_transformed[-3], col = c('red', 'blue')[as.numeric(as.factor(D_transformed$y))], pch = 20)

## Problem 12
SVM(train = D_train, kernel = kernel.quadratic)

## Example calls using the plotting feature
SVM(train = D_train, kernel = kernel.quadratic, plotApproximation = TRUE)

d <- data.generate(N = 150, generateTarget = TRUE)$data
tr <- d[1:50, ]
ts <- d[51:150, ]
SVM(train = tr, test = ts, kernel = NULL, plotApproximation = TRUE)
SVM(train = tr, test = ts, kernel = kernel.quadratic, plotApproximation = TRUE)
