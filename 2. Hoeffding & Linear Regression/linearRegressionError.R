lm.expectedE_in <- function(N, sigma, d) {
  (sigma^2)*(1 - (d+1)/N)
}

## Problem 1
lm.expectedE_in(10, 0.1, 8)
lm.expectedE_in(25, 0.1, 8)
lm.expectedE_in(100, 0.1, 8)
lm.expectedE_in(500, 0.1, 8)
lm.expectedE_in(1000, 0.1, 8)