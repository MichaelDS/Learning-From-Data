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