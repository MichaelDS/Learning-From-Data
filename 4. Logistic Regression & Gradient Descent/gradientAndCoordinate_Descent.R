############### GRADIENT DESCENT ###############
#
# Gradient descent is a general iterative method for nonlinear optimization.  It requires that the surface for
# which the minimum is being found be twice differentiable.  The procedure involves choosing a starting point
# and then taking fixed sized steps in the direction of the steepest slope until a stopping condition is met which,
# ideally, occurs when the minimum has been found.  It is typical for stopping conditions to attempt to detect when
# the slope along the gradient becomes flat, indicating a minimum.  For this reason, the step size can not be too large
# because it risks overshooting the minimum and bouncing across the sides of the surface.  On the other hand, a step
# size that is too small can be computationally expensive.  For this reason, a heuristic learning rate is sometimes 
# used in place of a step size, so that the steps scale with the slope along the gradient.  
#
# Suppose we are searching for the minimum on a surface E with respect to w.
# When using a fixed step size eta, the update at each step looks as follows:
#
# w(1) <- w(0) + eta*v, where v is the unit vector along the steepest slope at that point; thus,
# v = -gradient(E(w(0)))/||gradient(E(w(0)))||
#
# When using a fixed learning rate eta, the update at each step looks as follows:
#
# w(1) <- w(0) - eta*gradient(E(w(0)))
#
# When minimizing an error function using a training data set of size N, gradient(E) is based on all examples (xn, yn)
#
############### STOCHASTIC GRADIENT DESCENT ###############
#
# "Batch" gradient descent, as described above, is computationally expensive and susceptible to problematic artifacts
# of optimization such as local minima and flat regions.  These issues can be mitigated by using a randomized version
# of gradient descent.
#
# Stochastic gradient descent chooses one example (xn, yn) at a time from the training set at random and applies 
# gradient descent to it.  Although this only optimizes w with respect to a single example at a time, the 
# "average" direction that is descended along using this method is the same as in the "batch" version of gradient 
# descent.
#
# E[-gradient(h(xn), yn)] = (1/N)*SIGMA(n = 1, N) -gradient(h(xn, yn))
#                         = -gradient(E)
#
# Benefits of stochastic gradient descent include cheaper computation, randomization of "starting points" helps to
# avoid getting trapped in silly local minima and flat regions, and its sheer simplicity has motivated the 
# development of many heuristics; for example, a rule of thumb is that a learning rate of eta = 0.1 is generally
# satisfactory.

############### IMPLEMENTATION ###############

## Performs gradient descent on a specified surface with respect to specified parameters
## Returns a list containing the optimized parameter values, the value of the surface evaluated at these parameters, and the number of iterations taken by the algorithm
## expr - An expression representing the surface being minimized
## parameters - A vector of characters representing the parameters with respect to which expr is being optimized
## values - A vector of initial values for the parameters, in corresponding order to the parameters vector
## goal - A stopping condition
## maxIter - The maximum number of iterations to perform before terminating
## eta - The learning rate
gradient.descent <- function(expr, parameters, values, goal = quote(eval(expr <= -Inf)), maxIter = Inf, eta = 0.1) {
  for(i in 1:length(values)) {           # assign initial values to the appropriately named variables in this environment
    assign(parameters[i], values[[i]])
  }
  iterations <- 0
  repeat {                               # begin new epoch
    d_values <- -eta*attributes(eval(deriv(expr, parameters)))[[1]]  # compute the change in parameter values for this epoch; uses the deriv function to compute the gradient which is returned as an attribute in an object and extracted using the attributes function 
    values <- values + d_values                                      # update the parameter values
    for(i in 1:length(values)) {                                     # assign the new values to their respective variables
      assign(parameters[i], values[[i]])
    }
    iterations <- iterations + 1                                     # update iteration count
    if(eval(goal) || iterations >= maxIter)                          # check stopping conditions
      break
  }
  list(parameter_values = values, f = eval(expr), numIterations = iterations)
}

## Performs coordinate descent
## Implemented purely for comparison with gradient descent
## In each iteration, the algorithm moves in the direction of each of the coordinates, one at a time.
## expr - An expression representing the surface being minimized
## parameters - A vector of characters representing the parameters with respect to which expr is being optimized
## values - A vector of initial values for the parameters, in corresponding order to the parameters vector
## goal - A stopping condition
## maxIter - The maximum number of iterations to perform before terminating
## eta - The learning rate
coordinate.descent <- function(expr, parameters, values, goal = quote(eval(expr <= -Inf)), maxIter = Inf, eta = 0.1) {
  for(i in 1:length(values)) {         # assign initial values to the appropriately named variables in this environment
    assign(parameters[i], values[i])
  }
  iterations <- 0
  repeat {                             # begin new iteration
    for(i in 1:length(values)) {       # consider each coordinate individually
      d_value <- -eta*attributes(eval(deriv(expr, parameters[i])))[[1]]  # update the coordinate value using the gradient and learning rate
      values[i] <- values[i] + d_value                                   # store the new value
      assign(parameters[i], values[i])                                   # assign the new value to the appriate variable
    }
    iterations <- iterations + 1                                         # update iteration count
    if(eval(goal) || iterations >= maxIter)                              # check stopping conditions
      break
  }
  list(parameter_values = values, f = eval(expr), numIterations = iterations) 
}

## Problems 5 & 6
gradient.descent(quote((u*exp(v) - 2*v*exp(-u))^2), c('u', 'v'), data.frame(1,1), goal = quote(eval(expr) <= 10^-14))

## problem 7
coordinate.descent(quote((u*exp(v) - 2*v*exp(-u))^2), c('u', 'v'), c(1,1), maxIter = 15) 

## example using parse instead of quote
#gradient.descent(parse(text = '(u*exp(v) - 2*v*exp(-u))^2'), c('u', 'v'), data.frame(1,1), goal = 10^(-14))  
