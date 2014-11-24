############### SOFT-MARGIN SUPPORT VECTOR MACHINES ###############

############### NON-LINEAR CLASSIFICATION VIA THE KERNEL TRICK ###############




digits.SVM <- function(train, test, digits = c(1, 5), C = 0.01, kernel = 'radial', degree = 3, gamma = 1, coef0 = 0, scale = FALSE, type = 'C-classification', classification.plot = FALSE) {
  library(e1071)
  if(length(digits) != 1 && length(digits) != 2)
    stop('Invalid length of digits vector.  Must specify one or two digits to classify')
  if(length(digits) == 2) {
    train <- train[(train$digit == digits[1]) | (train$digit == digits[2]), ]
    test <- test[(test$digit == digits[1]) | (test$digit == digits[2]), ]
  }
    
  train$class <- -1
  test$class <- -1
  train[train$digit == digits[1], ]$class <- 1
  test[test$digit == digits[1], ]$class <- 1
  fit <- svm(class~intensity + symmetry, data = train, cost = C, kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0, scale = scale, type = type)
  train$fitted <- predict(fit, train[c('intensity', 'symmetry')])
  test$fitted <- predict(fit, test[c('intensity', 'symmetry')])
  
  if(classification.plot) {
    gridRange <- apply(train[c('intensity', 'symmetry')], 2, range)
    x1 <- seq(from = gridRange[1, 1] - 0.025, to = gridRange[2, 1] + 0.025, length = 75)
    x2 <- seq(from = gridRange[1, 2] - 0.05, to = gridRange[2, 2] + 0.05, length = 75)
    grid <- expand.grid(intensity = x1, symmetry = x2)
    grid$class <- predict(fit, grid)
    decisionValues <- predict(fit, grid, decision.values = TRUE)
    grid$z <- as.vector(attributes(decisionValues)$decision)
    
    ##  TESTING PURPOSES  
    #   print(range(grid$z))
    #   print(sum(train$fitted == -1))
    #   print(length(train$fitted))
    
    ##  GGPLOT VERSION OF PLOT; CONTOUR NEEDS DEBUGGING  
    #   library(ggplot2)
    #   p <- ggplot(data = grid, aes(intensity, symmetry, colour = as.factor(class))) + 
    #     geom_point(size = 1.5) +
    #     scale_fill_manual(values = c('red', 'black')) + 
    #     stat_contour(data = grid, aes(x = intensity, y = symmetry, z = z), breaks = c(0)) + 
    #     geom_point(data = train, aes(intensity, symmetry, colour = as.factor(class)), alpha = 0.7) +
    #     scale_colour_manual(values = c('red', 'black')) + labs(colour = 'Class') +
    #     scale_x_continuous(expand = c(0,0)) +
    #     scale_y_continuous(expand = c(0,0))
    #   print(p)
    
    par(mfrow = c(1,2))
    ## Note: RGB Specification seems to increase running and plotting time complexity
    plot(grid[c('intensity', 'symmetry')], col = ifelse(grid$class == 1, '#0571B070', '#CA002070'), main = 'Training', pch='20', cex=.2)
    points(train[c('intensity', 'symmetry')], col = ifelse(train$class == 1, '#0571B070', '#CA002070'))
    contour(x1, x2, matrix(grid$z, length(x1), length(x2)), level=0, lwd = 1.5, drawlabels = FALSE, add=TRUE)
    plot(grid[c('intensity', 'symmetry')], col = ifelse(grid$class == 1, '#0571B070', '#CA002070'), main = 'Test', pch='20', cex=.2)
    points(test[c('intensity', 'symmetry')], col = ifelse(test$class == 1, '#0571B070', '#CA002070'))
    contour(x1, x2, matrix(grid$z, length(x1), length(x2)), level=0, lwd = 1.5, drawlabels = FALSE, add=TRUE)
    mtext(paste('Digit Classification Plots:', digits[1], 'vs', ifelse(length(digits) == 2, digits[2], 'All'), '\nKernel:', kernel, '\nC:', C), line = -3, outer = TRUE)
  }

  list(E_in = mean(train$fitted != train$class), E_out = mean(test$fitted != test$class), num_support_vectors = nrow(fit$SV))
}

digits.SVM.CVcost <- function(train, digits = c(1, 5), k = 10, numTrials = 100, C = c(0.0001, 0.001, 0.01, 0.1, 1), kernel = 'radial', degree = 3, gamma = 1, coef0 = 0, scale = FALSE, type = 'C-classification', classification.plot = FALSE) {
  library(caret)
  bestC.count <- numeric(length(C))
  E_cv <- numeric(length(C))
  for(i in 1:numTrials) {
    groups <- createFolds(train$digit)
    E_val <- numeric(length(C))
    for(j in 1:length(C)) {
      for(p in 1:length(groups)) {
        D_train <- train[-groups[[p]], ]
        D_val <- train[groups[[p]], ]
        E_val[j] <- E_val[j] + digits.SVM(D_train, D_val, digits = digits, C = C[j], kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0, scale = scale, type = type, classification.plot = classification.plot)$E_out
      }
    }
    bestC <- which(E_val == min(E_val))
    if(length(bestC) > 1)                                             # if there is a tie in E_cv, select the smaller C
      bestC <- min(bestC)
    bestC.count[bestC] <- bestC.count[bestC] + 1
    E_cv <- E_cv + E_val/k
  }
  E_cv <- E_cv/numTrials
  bestC.index = which(bestC.count == max(bestC.count))
  list(best_C = C[bestC.index], E_cv = E_cv[bestC.index])
}

train <- read.table('train.txt', col.names = c('digit', 'intensity', 'symmetry'))
test <- read.table('test.txt', col.names = c('digit', 'intensity', 'symmetry'))

## Problem 2
digits.SVM(train, test, digits = c(0), kernel = 'polynomial', degree = 2, coef0 = 1) # 0 vs All
digits.SVM(train, test, digits = c(2), kernel = 'polynomial', degree = 2, coef0 = 1) # 2 vs All
digits.SVM(train, test, digits = c(4), kernel = 'polynomial', degree = 2, coef0 = 1) # 4 vs All
digits.SVM(train, test, digits = c(6), kernel = 'polynomial', degree = 2, coef0 = 1) # 6 vs All
digits.SVM(train, test, digits = c(8), kernel = 'polynomial', degree = 2, coef0 = 1) # 8 vs All

## Problem 3
digits.SVM(train, test, digits = c(1), kernel = 'polynomial', degree = 2, coef0 = 1) # 1 vs All
digits.SVM(train, test, digits = c(3), kernel = 'polynomial', degree = 2, coef0 = 1) # 3 vs All
digits.SVM(train, test, digits = c(5), kernel = 'polynomial', degree = 2, coef0 = 1) # 5 vs All
digits.SVM(train, test, digits = c(7), kernel = 'polynomial', degree = 2, coef0 = 1) # 7 vs All
digits.SVM(train, test, digits = c(9), kernel = 'polynomial', degree = 2, coef0 = 1) # 9 vs All

## Problem 4
numSV_0vsAll <- digits.SVM(train, test, digits = c(0), kernel = 'polynomial', degree = 2, coef0 = 1)$num_support_vectors
numSV_1vsAll <- digits.SVM(train, test, digits = c(1), kernel = 'polynomial', degree = 2, coef0 = 1)$num_support_vectors
numSV_0vsAll - numSV_1vsAll 

## Problems 5 & 6
digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 3, coef0 = 1, C = 0.0001)
digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 3, coef0 = 1, C = 0.001)
digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 3, coef0 = 1, C = 0.01)
digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 3, coef0 = 1, C = 0.1)
digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 3, coef0 = 1, C = 1)

digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 5, coef0 = 1, C = 0.0001)
digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 5, coef0 = 1, C = 0.001)
digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 5, coef0 = 1, C = 0.01)
digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 5, coef0 = 1, C = 0.1)
digits.SVM(train, test, digits = c(1, 5), kernel = 'polynomial', degree = 5, coef0 = 1, C = 1)

## Problems 7 & 8
digits.SVM.CVcost(train, digits = c(1, 5), kernel = 'polynomial', degree = 2, coef0 = 1)

## Problems 9 & 10
digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 0.01)
digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 1)
digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 100)
digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 10^4)
digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 10^6)

## Sample calls digits.SVM with the plotting feature activated
# digits.SVM(train, test, digits = c(0), kernel = 'polynomial', degree = 2, coef0 = 1, classification.plot = TRUE)
# digits.SVM(train, test, digits = c(1), kernel = 'polynomial', degree = 2, coef0 = 1, classification.plot = TRUE)
# digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 0.01, classification.plot = TRUE)
# digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 1, classification.plot = TRUE)
# digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 100, classification.plot = TRUE)
# digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 10^4, classification.plot = TRUE)
# digits.SVM(train, test, digits = c(1, 5), kernel = 'radial', gamma = 1, C = 10^6, classification.plot = TRUE)