############### APPROXIMATION-GENERALIZATION TRADEOFF ###############
#
# All learning problems involve an approximation-generalization tradeoff in that models with more complex hypothesis 
# sets have a better chance of approximating their target functions while models with less complex hypothesis sets
# have a better chance of generalizing out of sample. One method of quantifying this tradeoff is VC analysis which
# imposes the following bound on out-of-sample error:
#
# With probability >= 1 - delta, where delta = right-hand side of the VC inequality;
# P[|E_in(g) - E_out(g)| > epsilon] <= 4*m_H(2N)*e^(-(1/8)*(epsilon^2)*N) = delta
#
# E_out <= E_in + omega
#
# where omega = sqrt((8/N)*ln(4*m_H(2N)/delta)), N = sample size, 
# m_H(N) = growth function = SIGMA(i = 0, d_VC)Choose(N, i), and d_VC = VC Dimension = the largest number of points a
# hypothesis set can shatter, or the largest value of N for which m_H(N) = 2^N.  
#
# The number of parameters in a model constitute analog degrees of freedom; d_VC translates these into
# binary degrees of freedom for the context of producing dichotomies instead of continuous values. These quantities are
# abstract measures of the expressiveness of a model.  Note that d_VC measures the EFFECTIVE number of parameters.
# A rule of thumb is that a sample size >= 10*d_VC should be used in order to ensure that the VC inequality
# produces meaningful bounds.
#
# Another way of quantifying the tradeoff is bias-variance analysis.  It applies to real-valued targets and 
# decomposes E_out as follows:
#
# The expected value of of E_out with respect to all possible training sets of size N, 
# D = (x1, y1), (x2, y2), ..., (x_N, y_N), where the x_n's are picked independently from X, is the sum of the model's 
# expected bias and expected variance (and expected energy of the stochastic noise in the case of noisy targets); 
# Expectation_D[E_out] = Expectation_x[bias] + Expectation(D,x)[variance] (+Expectation_(eps,x)[eps(x)^2].  Bias is a 
# measure of the best approximation error that the model can achieve; it does not depend on D and is determined by the 
# (in)ability of the model to replicate the target as well as by the variance of the noise (note the distinction 
# between this variance and the variance of the model's performance, which is the subject of bias-variance analysis).  
# The variance of the model's performance is a measure of how much the model's approximation varies from it's best 
# performance; this measure does depend on D.  In order to optimize the bias-variance tradeoff, one should always 
# match the model complexity to the data resources, NOT the target complexity.  
#
############### EXPERIMENT ###############
#
# Learning curves illustrate how in-sample and out-of-sample error vary with the size of the data set, N.  Their plots
# indicate how well E_in generalizes to E_out across a range of N and, thus, can reveal local optima with respect to
# values of N producing hypotheses with strong out-of-sample performance.  
#
# This simulation implements linear regression and uses it to approximate a specified real-valued target across various
# datasets, D, across various values of N.  E_in and E_out are measured during each run in order to aggregate the
# results and use them to construct learning curves.  Each run generates a dataset D with dimension specified in 
# the parameters to the simulation function.  Excluding the threshold value, each vector x in X is made up of 
# uniformly distributed real-numbered values between 0 and a specified maximum.  The response value for each vector
# is produced by applying the target function to the X and then adding noise in the form of normally distributed
# values with a mean of 0 and a specified standard deviation.  Linear regression is then applied to this data in order
# to produce a final hypothesis and the results of the hypothesis are compared to the response using ordinary least
# squares in order caculate E_in.  In order to calculate E_out, the same X values are used and the response values are
# produced again using fresh noise; this allows this data set to play the role of an out-of-sample data set.  
#
# The learning curves are plotted twice but with different shadings; one emphasizing in-sample error and generalization
# error and the other emphasizing error due to model bias and variance.  The curves will always exhibit an asymptotic
# tendency toward the bias, which, in this case, is equal to the variance of the errors because the model is 
# capable of fully replicating the target function.  

############### IMPLEMENTATION ###############

## Function for generating data
data.generate <- function(n = 100, xdim = 2, xmax = 100) {
  X <- cbind(1, matrix(NA, n, xdim))
  names <- character(xdim + 1)
  names[1] <- 'x0'
  if(xdim == 0) {
    colnames(X) <- names
    return(X[,1])
  }
  for(i in 2:(xdim + 1)) {
    X[, i] <- runif(n, 0, xmax)
    names[i] <- paste('x', i - 1, sep = '')
  }
  colnames(X) <- names
  X
}

## Plots learning curves using the average values of E_in and E_out from a specified number of trials
## Uses linear regression to approximate the target function specified by w_target across specified values of N
## xdim - The dimensionality (and degrees of freedom) of the data 
## xmax = The upper threshold for x values in the data set 
## noisiness - Model bias; the standard deviation of the noise 
learningCurves.simulate <- function(numTrials = 1000, noisiness = 5, xdim = 10, w_target = runif(xdim + 1, 0, 10), N_max = 1000, N_step = 50, xmax = 100) {
  N <- seq(xdim+1, N_max, N_step) # Initialize vector of sample sizes to test; start at d_VC = d + 1
  e_in <- numeric(length(N))      
  e_out <- numeric(length(N))
  for(i in 1:numTrials) {
    for(i in 1:length(N)) {
      X <- data.generate(n = N[i], xdim = xdim, xmax = xmax)                    # generate a data set
      y_in <- as.vector(X%*%w_target) + rnorm(N[i], mean = 0, sd = noisiness)   # apply the target function and add normally distributed noise to generate the target response values
      w <- solve(t(X)%*%X)%*%t(X)%*%y_in                # use linear regression to determine the final hypothesis
      y_model <- as.vector(X%*%w)                       # apply the hypothesis to the data to produce in-sample approximations
      e_in[i] <- e_in[i] + mean((y_model - y_in)^2)     # use OLS to calculate e_in and aggregate it for averaging later
      y_out <- as.vector(X%*%w_target) + rnorm(N[i], mean = 0, sd = noisiness)  # produce out-of-sample approximations using the same data set and fresh noise 
      e_out[i] <- e_out[i] + mean((y_model - y_out)^2)  # use OLS to calculate e_out and aggregate it for averaging later
    }
  }

  library(ggplot2)
  library(gridExtra)
  
  # take the vectorized average of the error measure by sample size vectors by dividing by the number of trials
  # create a data frame containing the resulting vectors, the corresponding vector of N values, and bias and label values for use with ggplot2
  e <- data.frame(N, e_in = e_in/numTrials, e_out = e_out/numTrials, bias = noisiness^2, label = 'sigma^2')  

  base <- ggplot(e) + stat_smooth(aes(x = N, y = e_in, colour = 'E_in'), method = 'loess', se = FALSE) +
  stat_smooth(aes(x = N, y = e_out, colour = 'E_out'), method = 'loess', se = FALSE)
  
  gg1 <- ggplot_build(base) # build ggplot object in order to extract the y-values of the loess lines
  
  # construct a new data frame containing the y-values of the loess lines
  # this step is necessary for proper shading because loess uses best-fit methods to calculate y-values for the lines
  e2 <- data.frame(x = gg1$data[[1]]$x, ymin = gg1$data[[1]]$y, ymax = gg1$data[[2]]$y, bias = noisiness^2)

  plot1 <- base + geom_abline(intercept = noisiness^2, slope = 0) + geom_text(aes(0, bias, label = label, vjust = -0.3, hjust = 0.5), parse = TRUE) +
  geom_ribbon(data = e2, aes(x = x, ymin = ymin, ymax = ymax, fill = 'Generalization Error'), alpha = 0.3) +
  geom_ribbon(data = e2, aes(x = x, ymin = 0, ymax = ymin, fill = 'In-Sample Error'), alpha = 0.3) +
  scale_fill_manual(values = c('red', 'blue')) + theme(legend.title = element_blank()) + 
  scale_color_manual(values = c('blue', 'red')) + ylab('Expected Error') + ggtitle('In-Sample Error & Generalization Error')
  
  plot2 <- base + geom_abline(intercept = noisiness^2, slope = 0) + geom_text(aes(0, bias, label = label, vjust = -0.3, hjust = 0.5), parse = TRUE) +
    geom_ribbon(data = e2, aes(x = x, ymin = bias, ymax = ymax, fill = 'Variance'), alpha = 0.3) +
    geom_ribbon(data = e2, aes(x = x, ymin = 0, ymax = bias, fill = 'Bias'), alpha = 0.3) +
    scale_fill_manual(values = c('blue', 'red')) + theme(legend.title = element_blank()) +
    scale_color_manual(values = c('blue', 'red')) + ylab('Expected Error') + ggtitle('Error due to Bias & Variance')
  
  grid.arrange(plot1, plot2, ncol=2)
}

set.seed(10111)
learningCurves.simulate()
