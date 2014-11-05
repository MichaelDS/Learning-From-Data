# Consider a noisy target y = t(w)%*%x + eps, where x belongs to R^(d) (with the added coordinate x0 = 1), 
# y belongs to R, w is an unknown vector, and eps is a noise term with zero mean and sigma^2 variance. Assume eps is 
# independent of x and of all other eps's. If linear regression is carried out using a training data set 
# D = {(x1; y1), ..., (xN; yN)}, and outputs the parameter vector w_lin, it can be shown that the expected 
# in-sample error E_in with respect to D is given by:
#
# E_D[E_in(w_lin)] = sigma^2*(1 - (d + 1)/N), where E_D denotes expectation with respect to D

## Calculates expected in-sample error
lm.expectedE_in <- function(N, sigma, d) {
  (sigma^2)*(1 - (d+1)/N)
}

## Problem 1
lm.expectedE_in(10, 0.1, 8)
lm.expectedE_in(25, 0.1, 8)
lm.expectedE_in(100, 0.1, 8)
lm.expectedE_in(500, 0.1, 8)
lm.expectedE_in(1000, 0.1, 8)