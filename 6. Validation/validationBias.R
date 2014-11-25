############### VALIDATION BIAS ###############

## Computes the expected values of e1, e2, and min(e1, e2) by Monte-Carlo method
## N - Number of examples to use
validationBias.monteCarlo <- function(N = 10000) {
  e1 <- runif(N, 0, 1)
  e2 <- runif(N, 0, 1)
  e <- numeric(N)
  for(i in 1:N)
    e[i] <- min(e1[i], e2[i])
  list(expected_e1 = mean(e1), expected_e2 = mean(e2), expected_minimum_of_e1_and_e2 = mean(e))
}

validationBias.monteCarlo() # Problem 6
