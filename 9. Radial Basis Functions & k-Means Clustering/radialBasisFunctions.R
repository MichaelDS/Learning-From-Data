## Generates the input data and uses the target function f(x) = sign(x2 - x1 + 0.25*sin(pi*x1)) to generate response values
## Ensures that the response values do not all belong to a single class
data.generate <- function(N = 100, limits = c(-1, 1)) { 
  repeat{
    # generate the points
    x1 <- runif(N, limits[1], limits[2])
    x2 <- runif(N, limits[1], limits[2])
    
    # compute responses
    y <- sign(x2 - x1 + 0.25*sin(pi*x1))
    
    if(abs(sum(y)) != length(y))  # make sure the points are not all of the same class
      break
  } 
  data.frame(x1, x2, y) # put the training set together and return it
}

## Can also use R's dist() function
## Can be used to find the Euclidean norm of a vector (setting on argument to 0) instead of using R's norm() function
dist.euclidian <- function(x1, x2) {
  sqrt(sum((as.vector(x1) - as.vector(x2))^2))  # as.vector() avoids potentially having to deal with dimension conformability
}

## X - A data frame of points to cluster
## Returns a matrix containing the coordinates of the cluster centers and a vector of cluster indices corresponding to the input data
cluster.kMeans <- function(X, nCenters = 9, clusters.plot = FALSE) {
  # initialize cluster centers using random sample of points
  # centers do not have to be equal to any points; this is an artifact of initialization.  The centers will shift as the algorithm iterates
  centers <- X[sample(1:nrow(X), nCenters), ]  
  X$cluster <- 0
  
  f <- function(i, j) dist.euclidian(X[i, -ncol(X)], centers[j,])  # define function to be passed to outer

  repeat {
    X$cluster <- apply(outer(1:nrow(X), 1:nrow(centers), Vectorize(f)), 1, function(x) which(x == min(x)))  # cluster each point according to the nearest center; use outer() to generate a points vs centers distance matrix, then choose the nearest center using apply() and which()

#     for(i in 1:nrow(X)) {                                           # vectorized version using outer is more efficient
#       minDist <- Inf
#       for(j in 1:nrow(centers)) {
#         d <- dist.euclidian(X[i, -ncol(X)], centers[j, ])
#         #d <- as.vector(dist(rbind(X[i, -ncol(X)], centers[j, ])))  # avoided rbinding and typecasting by writing a simple distance function
#         if(d < minDist) {
#           minDist <- d
#           X[i, ]$cluster <- j
#         }
#       }
#     }
    
    if(0 %in% sapply(1:nCenters, function(x) length(which(X$cluster == x)))) {  # check if there are any empty clusters
      centers <- X[sample(1:nrow(X), nCenters), -ncol(X)]                       # reset centers
      X$cluster <- 0                                                            # reset cluster assignments
      next                                                                      # restart the algorithm
    }
    newCenters <- matrix(0, nrow(centers), ncol(centers), dimnames = list(c(), c('x1', 'x2')))
    for(i in 1:nrow(centers)) {                                                 # recalculate centers
      newCenters[i, ] <- colMeans(X[which(X$cluster == i), -ncol(X)])
    }
    if(identical(centers, newCenters))  # centers becomes a matrix with no labels (as opposed to a data frame) after first iteration, so identical can correctly compare it with newCenters
      break
    centers <- newCenters
  }
  
  if(clusters.plot) {
    library(ggplot2)
    library(ggthemes)
    centersDF <- data.frame(x1 = centers[, 'x1'], x2 = centers[, 'x2'], cluster = seq(1:nrow(centers)))
    p <- ggplot(data = X, aes(x1, x2, colour = as.factor(cluster))) + geom_point() +
      scale_colour_tableau() + theme_solarized(light = 'false') +
      labs(colour = 'cluster') + ggtitle('Clustered Data') +
      geom_point(data = centersDF, aes(colour = as.factor(cluster)), shape = 8, size = 2.5) +
      guides(colour = guide_legend(override.aes = list(size=2, shape = 16)))
      #scale_color_brewer(palette = 'Pastel1')   # decided to go with Tableau colors and dark solarized theme instead of a brewer set
    print(p)
  }
  
  list(centers = centers, clusters = X$cluster)
}

RBF <- function(train, test = NULL, gamma = 1.5, nCenters = 9, clustering = cluster.kMeans, classification.plot = FALSE) {
  X <- train[, -ncol(train)]
  centers <- clustering(X, nCenters)$centers   # takes a data frame as its first argument
  phi <- matrix(0, nrow(X), nrow(centers))
  X <- as.matrix(X)                            # casting to matrix so that row vectors are already aligned when passed to norm()
  
  f <- function(i, j) norm(as.matrix(X[i, ]) - as.matrix(centers[j, ]), type = 'F')
  vf <- Vectorize(f)
  phi <- outer(1:nrow(X), 1:nrow(centers), vf)

#   for(i in 1:nrow(phi)) {                    # vectorized version using outer() is more efficient
#     for(j in 1:ncol(phi)) {
#       phi[i, j] <- norm(as.matrix(X[i, ]) - as.matrix(centers[j, ]), type = 'F')
#     }
#   }

  phi <- cbind(1, exp(-gamma*phi^2))           # the column of 1's corresponds to the bias term (w0); can be thought of as an arbitrary center with gamma = 0
  w <- solve(t(phi)%*%phi)%*%t(phi)%*%train$y  # solve for w that minimizes the RBF's squared error for the given centers on the training set using the pseudo-inverse; one-step learning

  y_fit <- sign(t(w)%*%t(phi))                 # apply w to phi to get the fitted y values; w was trained on phi, not X
  E_in = mean(y_fit != train$y)
  
  E_out <- NA
  if(!is.null(test)) {
    X <- as.matrix(test[, -ncol(train)])
    phi <- outer(1:nrow(X), 1:nrow(centers), vf)
    phi <- cbind(1, exp(-gamma*phi^2))
    y_fit <- sign(t(w)%*%t(phi))
    E_out = mean(y_fit != test$y)
  }
  
  if(classification.plot) {
    gridRange <- apply(train[c('x1', 'x2')], 2, range)
    x1 <- seq(from = gridRange[1, 1] - 0.05, to = gridRange[2, 1] + 0.05, length = 75)
    x2 <- seq(from = gridRange[1, 2] - 0.05, to = gridRange[2, 2] + 0.05, length = 75)
    grid <- expand.grid(x1 = x1, x2 = x2)             # not casting to matrix as with X previously; data frame is friendlier for plotting.  Instead, transpose its vectors as they are passed to norm()
    f2 <- function(i, j) norm(t(as.matrix(grid[i, ])) - as.matrix(centers[j, ]), type = 'F')  # transposed the first vector in the norm function in order to align it with the second vector
    vf2 <- Vectorize(f2)
    phi.grid <- outer(1:nrow(grid), 1:nrow(centers), vf2)
    phi.grid <- cbind(1, exp(-gamma*phi.grid^2))
    grid$z <- as.vector(t(w)%*%t(phi.grid))    # store the decision values
    grid$y <- sign(grid$z)  # translated them into response values
    
    if(!is.null(test))
      par(mfrow = c(1,2))
    
    plot(grid[c('x1', 'x2')], col = ifelse(grid$y == 1, 'blue', 'red'), main = ifelse(is.null(test), paste('Classification Plot: Radial Basis Function with K Centers', '\nGamma:', gamma), 'Training'), pch='20', cex=.2)
    points(train[c('x1', 'x2')], col = ifelse(train$y == 1, 'blue', 'red'))
    points(centers, col = 'black', pch = 10, cex = 1.3)
    points(centers, col = 'black', pch = 10, cex = 1)
    contour(x1, x2, matrix(grid$z, length(x1), length(x2)), level=0, lwd = 1.5, drawlabels = FALSE, add=TRUE)
    
    if(!is.null(test)) {
      plot(grid[c('x1', 'x2')], col = ifelse(grid$y == 1, 'blue', 'red'), main = 'Test', pch='20', cex=.2)
      points(test[c('x1', 'x2')], col = ifelse(test$y == 1, 'blue', 'red'))
      contour(x1, x2, matrix(grid$z, length(x1), length(x2)), level=0, lwd = 1.5, drawlabels = FALSE, add=TRUE)
      mtext(paste('Classification Plots: Radial Basis Function with K Centers', '\nGamma:', gamma), line = -2, outer = TRUE)
    }
    
  }
  
  list(E_in = E_in, E_out = E_out)          
}

SVM <- function(train, test = NULL, C = 10^6, kernel = 'radial', degree = 3, gamma = 1.5, coef0 = 0, scale = FALSE, type = 'C-classification', classification.plot = FALSE) {
  library(e1071)
  
  fit <- svm(y ~ x1 + x2, data = train, cost = C, kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0, scale = scale, type = type)
  train$fitted <- predict(fit, train[c('x1', 'x2')])
  E_out <- NA
  if(!is.null(test)) {
    test$fitted <- predict(fit, test[c('x1', 'x2')])
    E_out <- mean(test$fitted != test$y)
  }
  
  if(classification.plot) {
    gridRange <- apply(train[c('x1', 'x2')], 2, range)
    x1 <- seq(from = gridRange[1, 1] - 0.05, to = gridRange[2, 1] + 0.05, length = 75)
    x2 <- seq(from = gridRange[1, 2] - 0.05, to = gridRange[2, 2] + 0.05, length = 75)
    grid <- expand.grid(x1 = x1, x2 = x2)
    grid$y <- predict(fit, grid)
    decisionValues <- predict(fit, grid, decision.values = TRUE)
    grid$z <- as.vector(attributes(decisionValues)$decision)
    
    if(!is.null(test))
      par(mfrow = c(1,2))
    
    plot(grid[c('x1', 'x2')], col = ifelse(grid$y == 1, 'black', 'red'), main = ifelse(is.null(test), paste('SVM Classification Plot -', kernel, 'kernel',  '\nGamma:', gamma, '\nC:', C), 'Training'), pch='20', cex=.2)
    points(train[c('x1', 'x2')], col = ifelse(train$y == 1, 'black', 'red'))
    sv.y <- train$y[fit$index]
    points(fit$SV, col = ifelse(sv.y == 1, 'black', 'red'), pch = 20)
    contour(x1, x2, matrix(grid$z, length(x1), length(x2)), level=0, lwd = 1.5, drawlabels = FALSE, add=TRUE)
    
    if(!is.null(test)) {
      plot(grid[c('x1', 'x2')], col = ifelse(grid$y == 1, 'black', 'red'), main = 'Test', pch='20', cex=.2)
      points(test[c('x1', 'x2')], col = ifelse(test$y == 1, 'black', 'red'))
      contour(x1, x2, matrix(grid$z, length(x1), length(x2)), level=0, lwd = 1.5, drawlabels = FALSE, add=TRUE)
      mtext(paste('Classification Plots: Support Vector Machine', '\nKernel:', kernel, '\nGamma:', gamma, '\nC:', C), line = -4, outer = TRUE)
    }
    
  }
  list(E_in = mean(train$fitted != train$y), E_out = E_out, num_support_vectors = nrow(fit$SV))
}

SVM.propInseparable <- function(numTrials = 1000, C = 10^6, kernel = 'radial', degree = 3, gamma = 1.5, coef0 = 0, scale = FALSE, type = 'C-classification', generator = data.generate, classification.plot = FALSE) {
  countInseparable <- 0
  for(i in 1:numTrials) {
    D <- generator()
    E_in <- SVM(D, C = C, kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0, scale = scale, type = type, classification.plot = classification.plot)$E_in
    if(E_in != 0)
      countInseparable <- countInseparable + 1
  }
  countInseparable/numTrials
}

SVM.simulate <- function(N_train = 100, N_test = 1000, numTrials = 100, C = 10^6, kernel = 'radial', degree = 3, gamma = 1.5, coef0 = 0, scale = FALSE, type = 'C-classification', generator = data.generate, classification.plot = FALSE) {
  E_in <- numeric(numTrials)
  E_out <- numeric(numTrials)
  num_SV <- numeric(numTrials)
  for(i in 1:numTrials) {
    train <- generator(N_train)
    test <- generator(N_test)
    res <- SVM(train, test, C = C, kernel = kernel, degree = degree, gamma, coef0 = coef0, scale = scale, type = type, classification.plot = classification.plot)
    E_in[i] <- res$E_in
    E_out[i] <- res$E_out
    num_SV[i] <- res$num_support_vectors
  }
  list(E_in = mean(E_in), E_out = mean(E_out), avg_num_support_vectors = mean(num_SV))
}

SVM_RBF.compare <- function(N_train = 100, N_test = 1000, numTrials = 100, nCenters = 9, C = 10^6, kernel = 'radial', degree = 3, gamma = 1.5, coef0 = 0, scale = FALSE, type = 'C-classification', clustering = cluster.kMeans, generator = data.generate, classification.plot = FALSE) {
  count <- 0
  for(i in 1:numTrials) {
    cat(paste('Iteration', i, '\n'))
    train <- generator(N_train)
    test <- generator(N_test)
    SVM_error <- SVM(train, test, C, kernel, degree, gamma, coef0, scale, type, classification.plot)$E_out
    RBF_error <- RBF(train, test, gamma, nCenters, clustering, classification.plot)$E_out
    if(SVM_error < RBF_error)
      count <- count + 1
  }
  count/numTrials
}

RBF.compareParameters <- function(N_train = 100, N_test = 1000, numTrials = 100, gamma = c(1.5), nCenters = c(9, 12), clustering = cluster.kMeans, generator = data.generate, classification.plot = FALSE) {
  if(!(length(nCenters) %in% 0:2) | !(length(gamma) %in% 0:2))
    stop('Invalid number of gamma and/or nCenters values. Must provide one or two values for each.')
  
  nCenters1 <- nCenters[1]
  if(length(nCenters) == 2)
    nCenters2 <- nCenters[2]
  else
    nCenters2 <- nCenters1
  gamma1 <- gamma[1]
  if(length(gamma) == 2)
    gamma2 <- gamma[2]
  else
    gamma2 <- gamma1
  
  count_Ein.up_Eout.down <- 0
  count_Ein.down_Eout.up <- 0
  count_both.up <- 0
  count_both.down <- 0
  count_no.change <- 0
  
  for(i in 1:numTrials) {
    cat(paste('Iteration', i, '\n'))
    train <- generator(N_train)
    test <- generator(N_test)
    res1 <- RBF(train, test, gamma1, nCenters1, clustering, classification.plot)
    res2 <- RBF(train, test, gamma2, nCenters2, clustering, classification.plot)
    if(res2$E_in > res1$E_in & res2$E_out < res1$E_out)
      count_Ein.up_Eout.down <- count_Ein.up_Eout.down + 1
    else if(res2$E_in < res1$E_in & res2$E_out > res1$E_out)
      count_Ein.down_Eout.up <- count_Ein.down_Eout.up + 1
    else if(res2$E_in > res1$E_in & res2$E_out > res1$E_out)
      count_both.up <- count_both.up + 1
    else if(res2$E_in < res1$E_in & res2$E_out < res1$E_out)
      count_both.down <- count_both.down + 1
    else if(res2$E_in == res1$E_in & res2$E_out == res1$E_out)
      count_no.change <- count_no.change + 1
  }
  list(count_Ein.up_Eout.down = count_Ein.up_Eout.down, count_Ein.down_Eout.up = count_Ein.down_Eout.up, count_both.up = count_both.up, count_both.down = count_both.down, count_no.change = count_no.change)
}

RBF.simulate <- function(N_train = 100, N_test = 1000, numTrials = 100, gamma = 1.5, nCenters = 9, clustering = cluster.kMeans, generator = data.generate, classification.plot = FALSE) {
  E_in <- numeric(numTrials)
  E_out <- numeric(numTrials)
  count_perfectSeparation <- 0
  for(i in 1:numTrials) {
    cat(paste('Iteration', i, '\n'))
    train <- generator(N_train)
    test <- generator(N_test)
    res <- RBF(train, test, gamma, nCenters, clustering, classification.plot = classification.plot)
    E_in[i] <- res$E_in
    E_out[i] <- res$E_out
    if(res$E_in == 0)
      count_perfectSeparation <- count_perfectSeparation + 1
  }
  list(E_in = mean(E_in), E_out = mean(E_out), proportion_Ein_zero = count_perfectSeparation/numTrials)
}

## Problem 13
SVM.propInseparable(numTrials = 100, C = 10^6, gamma = 1.5)

## Problem 14
SVM_RBF.compare(nCenters= 9, gamma = 1.5, numTrials = 100)   # relatively long running time

## Problem 15
SVM_RBF.compare(nCenters= 12, gamma = 1.5, numTrials = 100)  # relatively long running time

## Problem 16
RBF.compareParameters(nCenters = c(9, 12), gamma = c(1.5), numTrials = 100) # relatively long running time

## Problem 17
RBF.compareParameters(nCenters = c(9), gamma = c(1.5, 2), numTrials = 100) # relatively long running time

## Problem 18
RBF.simulate(nCenters = 9, gamma = 1.5, numTrials = 100)

train <- data.generate(100)
test <- data.generate(1000)
  
## Re-run this various times to see SVM applied to various data sets
## Vary other parameters on the same data set to see their effects  
SVM(train, C = 10^6, gamma = 1.5, classification.plot = TRUE)
SVM(train, test, C = 10^6, gamma = 1.5, classification.plot = TRUE)

RBF(train, classification.plot = TRUE)
RBF(train, test, classification.plot = TRUE)
