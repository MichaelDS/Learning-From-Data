############### BIAS & VARIANCE ###############
#
# Bias-variance analysis is a method for quantifying approximation-generalization tradeoff that is particularly 
# well suited for analyzing the approximation of real-values targets.  It decomposes E_out as follows: 
#
# The expected value of of E_out with respect to all possible training sets of size N, 
# D = (x1, y1), (x2, y2), ..., (x_N, y_N), where the x_n's are picked independently from X, is the sum of the model's 
# bias and variance; Expectation_D[E_out] = bias + variance.  Bias is a measure of the best approximation error that 
# the model can achieve; it does not depend on D and is determined by the variance of the noise (note the distinction 
# between this variance and the variance of the model's performance, which is the subject of bias-variance analysis).
# The variance of the model's performance is a measure of how much the model's approximation varies from it's best
# performance; this measure does depend on D.  In order to optimize the bias-variance tradeoff, one should always
# match the model complexity to the data resources, NOT the target complexity.  
# 
############### DEFINITIONS ###############
#
# N = Sample Size
# f = The target function
# g = A hypothesis function for approximating a target function
# g^D = A hypothesis function for approximating a target function given a training set of size N, D
# E_D = Expectation with respect to all possible training sets of size N, D
# E_x = Expectation with respect to x
# E_in(g) = In-Sample Error
# E_out(g) = Out-of-Sample Error
#
# g_bar(x) = the "average" hypothesis = E_D[g^D(x)]
#
# 1. E_out(g^D) = E_x[(g^D(x) - f(x))^2]
#
# 2. E_D[E_out(g^D)] = E_D[E_x[(g^D(x) - f(x))^2]] = E_x[E_D[(g^D(x) - f(x))^2]]
#
# 3. E_D[(g^D(x) - f(x))^2] = E_D[(g^D(x) - g_bar(X))^2] + (g_bar(x) - f(x))^2 = var(x) + bias(x)
#
# 4. Therefore, E_D[E_out(g^D)] = E_x[E_D[(g^D(x) - f(x))^2]] = E_x[bias(x) + var(x)] = bias + var
#
# bias = (g_bar(x) - f(x))^2
# var = E_D[(g^D(x) - g_bar(X))^2]
#
############### EXPERIMENT ###############
#
# This simulation uses various linear models to approximate a target function f:[-1, 1] -> R given by f(x) = sin(pi*x) 
# with a uniform input probability distribution on [-1, 1].  Each run generates a training set with only two 
# independently examples.  The selected learning algorithm then picks the hypothesis that minimizes the mean squared 
# error on the examples.  This is repeated for a specified number of trials after which g_bar, the average out of 
# sample error, e_out, model bias, and model variance are calculated and returned. 
#
# The model using a line through the origin, h(x) = ax, has the best combination of model bias and variance 
# among the models tested for approximating f(x) = sin(pi*x).  The model using a line with a variable intercept, 
# h(x) = ax + b, has the best model bias, however, this is offset by its relatively high model variance.  
# More complex quadratic models have progressively worse model bias and variance.  This is because such models are too 
# complex for the data resources provided (only two training examples).  
#
# If specified, the model's bias and variance is illustrated by constructing a plot depicting the target sinusoid 
# function, f(x), the selected model's expected hypothesis, g_bar(x), and every final hypothesis produced by the model 
# throughout the simulation.  It is possible to see the consequences of the high variance of exhibited by the 
# quadratic models by making repeated calls to the simulation.  Either of the quadratic models produces very different 
# expected hypotheses for each call; the smaller the number of trials used, the more amplified this effect becomes.  
# In contrast, the simpler models tend to produce relatively consistent expected hypotheses.
#

############### IMPLEMENTATION ###############

## Generates input data and applies the target function, sin(pi*x), to it; returns the resulting training set
data.generate <- function(N = 2) {
  X <- runif(N, -1, 1)
  y <- sin(pi*X)
  cbind(X, y)
}

## Applies the learning model, h(x) = b, to provided training data
## If degree is set to TRUE, the function will return the degree of the model instead
## If equation is set to true, the function will return a string representation of the model instead
lm.horizontalLine <- function(X = NULL, y = NULL, degree = FALSE, equation = FALSE) {
  if(degree)
    return(0)
  else if(equation)
    return('h(x) = b')
  
  b <- matrix(1, length(X), 1)
  rbind(solve(t(b)%*%b)%*%t(b)%*%y, 0)
}

## Applies the learning model, h(x) = ax, to provided training data
lm.lineThroughOrigin <- function(X = NULL, y = NULL, degree = FALSE, equation = FALSE) {
  if(degree)
    return(1)
  else if(equation)
    return('h(x) = ax')
  
  rbind(0, solve(t(X)%*%X)%*%t(X)%*%y)
}

## Applies the learning model, h(x) = ax + b, to provided training data
lm.lineAndIntercept <- function(X = NULL, y = NULL, degree = FALSE, equation = FALSE) {
  if(degree)
    return(1)
  else if(equation)
    return('h(x) = ax + b')
  
  X <- cbind(1, X)
  solve(t(X)%*%X)%*%t(X)%*%y
}

## Applies the learning model, h(x) = ax^2, to provided training data
lm.quadraticThroughOrigin <- function(X = NULL, y = NULL, degree = FALSE, equation = FALSE) {
  if(degree)
    return(2)
  else if(equation)
    return(expression(h(x)~'='~a*x^2))
  
  rbind(0, solve(t(X^2)%*%X^2)%*%t(X^2)%*%y)
}

## Applies the learning model, h(x) = ax^2 + b, to provided training data
lm.quadraticAndIntercept <- function(X = NULL, y = NULL, degree = FALSE, equation = FALSE) {
  if(degree)
    return(2)
  else if(equation)
    return(expression(h(x)~'='~a*x^2 + b))
  
  X <- cbind(1, X^2)
  solve(t(X)%*%X)%*%t(X)%*%y
}

## Uses a specified learning model to approximate the target function using a specified number of training examples
## Uses a specified number of testing examples to approximate out-of-sample error
## Calculates the average weights (average hypothesis), expected out-of-sample error, model bias, and model variance and returns them
## numTrials - Number of trials for which to repeat the experiment
## N_train - The sample size of the training sets
## N_test - The sample size of the test sets
## model - The learning model to use
## plotApproximations - When set to TRUE, will construct a plot illustrating bias/variance for the model; setting this to TRUE will significantly increase the run-time of the simulation
approximate.sinusoid <- function(numTrials = 1000, N_train = 2, N_test = 1000, model = lm.lineThroughOrigin, plotApproximations = FALSE) {
  g <- matrix(0, numTrials, 2)    # initialize matrix to hold hypotheses calculated during each trial
  e_out <- numeric(1)             # initialize vector to aggregate out-of-sample error
  degree <- model(degree = TRUE)  # this will hold a flag indicating the degree of the model
  title <- model(equation = TRUE) # retrieve expression object to serve as part of the title for the plot

  for(k in 1:numTrials) {
    D <- data.generate(N_train)                           # generate training data
    g[k, ] <- model(D[, 'X'], D[, 'y'])                   # compute and store the final hypothesis for this trial
    D_test <- data.generate(N_test)                       # generate test data
    y_model <- t(g[k, ])%*%t(cbind(1, D_test[, 'X']))     # apply the hypothesis to the test data
    e_out <- e_out + mean((y_model - D_test[, 'y'])^2)    # aggregate e_out for averaging later on
  }
  g_bar <- signif(apply(g, 2, sum)/numTrials, digits = 3) # compute the expected hypothesis by averaging across trials
  e_out <- e_out/numTrials                                # compute the average out-of-sample error

  if(plotApproximations) {
    library(ggplot2)
    library(plyr)

    # check if the model is quadratic or not in order to pass the correct function to stat_function()
    # use alply to construct a list of stat_function() results, applied over all hypotheses in g
    # construct the plot and print it
    if(degree == 2) {
      h <- alply(g, 1, function(hyp) {stat_function(aes(y=0, colour = 'g'), fun = function(x) {hyp[2]*x^2 + hyp[1]}, alpha = 0.4)})
      h_bar <- stat_function(aes(y=0, colour = 'g_bar'), fun = function(x) g_bar[2]*x^2 + g_bar[1])
    }
    else {
      h <- alply(g, 1, function(hyp) {stat_function(aes(y=0, colour = 'g'), fun = function(x) {hyp[2]*x + hyp[1]}, alpha = 0.4)})
      h_bar <- stat_function(aes(y=0, colour = 'g_bar'), fun = function(x) g_bar[2]*x + g_bar[1])
    }

    plot1 <- ggplot() + xlim(c(-1, 1)) + ylim(c(-1.5, 1.5)) + 
    h + h_bar + stat_function(aes(y=0, colour = 'sin(pi*x'), fun = function(x) sin(pi*x)) + 
    scale_color_manual(values = c('grey', 'red', 'blue'), labels = c(expression(g^(D)*(x)), expression(bar(g)*(x)), expression(f(x)))) + theme(legend.title = element_blank()) +
    xlab('x') + ylab('y') + ggtitle(title)

    suppressWarnings(print(plot1))
  }

  bias <- integrate(function(x) (1/2)*((g_bar[2]*x + g_bar[1]) - sin(pi*x))^2, -1, 1) # compute model bias via numerical integration; can also be computed via analytical integration or Monte-Carlo simulation
                                                                                      # (1/2) is present as a coefficient in the integrand because the input probability is uniform on [-1, 1]; thus, P(x)dx = (1/2)dx
      
  variance <- numeric(1)                                  # initialize vector to aggregate model variance
  for(k in 1:nrow(g))                                     # iterate through stored hypotheses
    variance <- variance + mean((g[k, ]%*%t(cbind(1, D_test[, 'X'])) - g_bar%*%t(cbind(1, D_test[, 'X'])))^2)  # calculate expected value with respect to x for the kth hypothesis and aggregate it
  variance <- variance/nrow(g)                            # divide by the number of hypotheses to compute the average model variance

#  variance <- integrate(function(x) (1/2)*(x^2/nrow(g))*sum((g[, 2] - g_bar[2])^2), -1, 1) # model variance can also be computed by numerical integration; this example is for the model h(x) = ax
                                                                                            # for simplicity of implementation across different models, Monte-Carlo method was used instead          

  list(expected_hypothesis = g_bar, out_of_sample_error = e_out, bias = bias, variance = variance)  # function output
}

## Problem 7
approximate.sinusoid(model = lm.horizontalLine)
approximate.sinusoid(model = lm.lineThroughOrigin) # Problems 4, 5, and 6
approximate.sinusoid(model = lm.lineAndIntercept)
approximate.sinusoid(model = lm.quadraticThroughOrigin)
approximate.sinusoid(model = lm.quadraticAndIntercept)


## Sample calls to simulate with the plotting feature activated; using small numTrials to mitigate running time
# approximate.sinusoid(numTrials = 100, model = lm.horizontalLine, plotApproximations = TRUE)
# approximate.sinusoid(numTrials = 100, model = lm.lineThroughOrigin, plotApproximations = TRUE) 
# approximate.sinusoid(numTrials= 100, model = lm.lineAndIntercept, plotApproximations = TRUE)
# approximate.sinusoid(numTrials = 100, model = lm.quadraticThroughOrigin, plotApproximations = TRUE)
# approximate.sinusoid(numTrials = 100, model = lm.quadraticAndIntercept, plotApproximations = TRUE)