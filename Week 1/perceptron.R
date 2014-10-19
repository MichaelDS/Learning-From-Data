# The Perceptron is a learning algorithm for supervised classification on linearly separable data.  The algorithm
# is implemented as follows:
#
# Given training data, (x1, y1), (x2, y2), ... , (xn, yn);
# Initialize a weight vector, w, of all zeroes.  
# Use this initial hypothesis to classify the data as follows:
#
# h(x) = y_model = sign(t(w) * x)
#
# Randomly pick a misclassified point; sign(t(w) * xn) != yn
# Take the product of xn and yn and use it to update the weight vector
# 
# w = w + xn * yn
#
# Use the updated hypothesis to classify the data; if any points are still misclassified, choose one at random
# and repeat the algorithm until all points are classified correctly.
#
# The Perceptron learning algorithm can be modified to accomodated non-linearly separable data by keeping track
# of the weight vector that produced the best in-sample error and using it as the final set of weights after
# terminating the algorithm at a designated iteration.  This version of PLA is referred to as the Pocket algorithm.
#
# This simulation implements and examines supervised classification via the Perceptron (pocket) learnin algorithm.  
# A target function f and a dataset D in d = 2 are created.  X = [-1, 1] x [-1, 1] with uniform probability of picking 
# each x in X.  Each run chooses a random line in the plane as the target function by taking the line passing through 
# two random, uniformly distributed points in [-1, 1] x [-1, 1], where one side of the line maps to +1 and the other 
# maps to -1. The inputs, x_n, of the data set are chosen as random points (uniformly in X), and the target function is 
# evaluated on each x_n to get the corresponding output, y_n. PLA is then be used to learn from this data and 
# the final hypothesis is tested against an independently generated data set.  


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


##### PLA  #####

simulate.PLA <- function(N_train = 10, N_test = 1000, numTrials = 1000, maxIterations = Inf, simulation = data.generate) {
  iterations <- numeric(0)    # initializing the iteration and misclassification probability vectors
  probability <- numeric(0)
  
  # Number of times to repeat the experiment is specified by numTrials
  
  for (i in 1:numTrials){
    generated <- simulation(n = N_train, generateTarget = TRUE) # generating points (set n=10 or n=100) and target function
    input <- as.matrix(cbind(1, generated$data[c(1,2)])) # creating the input matrix
    
    w <- c(0,0,0)  # initializing the weight vector
    
    #res <- apply(input,1,function(x) t(w)%*%x)  # multiplying transpose of w with each row of input matrix to get initial hypothesis function
    res <- as.vector(input %*% w) #equivalent operation
    
    best <- list(sum(sign(res) != generated$data$y)/length(res), w)  # initializing list to keep track of the best in-sample error achieved so far and the weight vector that produced it
    
    k <- 0  # initializing iteration counter
    
    while (any(sign(res) != generated$data$y) && k < maxIterations) # as long as any of the elements of res do not match the true output, y, and the iterations threshold has not been reached
                                                                    # the PLA algorithm continues to iterate  
    {                                           
      #cat("Iteration:", k, "\n")
      misclassified <- which(sign(res) != generated$data$y)  # getting the indices of the points for which hypothesis is wrong
      ifelse (length(misclassified) == 1, n <- misclassified, n <- sample(misclassified,1))  # randomly choose one of these points
      w <- w + generated$data$y[n]*input[n, ]       # update the weights
      res <- apply(input, 1, function(x) t(w)%*%x)  # use new weights to update the hypothesis function
      e_in <- sum(sign(res) != generated$data$y)/length(res) # calculate in-sample error
      if (e_in < best[[1]]) {
        best <- list(e_in, w) # if a the current weight vector is better than the previous best, store it
      }
      k <- k+1          # increment iteration count
    }
    
    w <- best[[2]]      #selecting the best weight vector discovered by the algorithm
    
    iterations[i] <- k  # store the number of iterations needed in this run
    
    new.data <- simulation(N_test)  #  generating the test points in order to examine out-of-sample performance
    f <- as.numeric(new.data$x1 * generated$slope + generated$intercept > new.data$x2) * 2 - 1  # classifying points according to the true function f
    g <- as.numeric(new.data$x1 * (-w[2]/w[3]) - w[1]/w[3] > new.data$x2) * 2 - 1    # classifying points according to the hypothesised function g, using the 
                                                                                     # final weights provided by PLA            
    
    probability[i] <- sum(f != g)/N_test  # store the misclassification error from this run
  }
  
  # Plot the points and f and g functions from the last iteration (purely illustrative purposes)
  
  library(ggplot2)
  
  plot1 <- qplot(x1,x2,col= as.factor(y), data = generated$data) + geom_abline(intercept = generated$intercept, slope = generated$slope) +
    geom_abline(intercept = -w[1]/w[3], slope = -w[2]/w[3], col=3)
  
  print(plot1)
  
  # Final results: average of iterations and estimated misclassification probabilities
  list(iterations = mean(iterations), probability = mean(probability)) 
}

simulate.PLA(10)  # Problems 7 & 8

simulate.PLA(100) # Problems 9 & 10
