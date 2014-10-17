# Hoeffding's inequality provides an upper bound on the probability of the empirical mean, nu, of a sample 
# of N independent random variables deviating from its expected value, mu, by more than some amount, eps.  
# In the case of Bernoulli random variables, it can be mathematically stated as follows:
# 
# P[|nu - mu] > eps] <= 2e^(-2(eps^2)N)
# 
# A larger sample size, N, tightens the bound whereas a stricter restriction, eps, loosens it.


## Simulation for flipping 1000 virtual fair coins a designated number of times
## Focuses three coins; the first coin flipped, a randomly chosen coin, and the coin with the minimum frequency of heads
## Returns a list containing generated distributions of the fraction of heads obtained for the three respective coins
simulate.flips <- function(numTrials = 100000, numFlips = 10) {
  v1 <- numeric(0)
  vrand <- numeric(0)
  vmin <- numeric(0)
  for (i in 1:numTrials) {
    res <- rbinom(n = 1000, size = numFlips, prob = 0.5)
    v1[i] <- res[1]/numFlips
    vrand[i] <- sample(res, 1)/numFlips
    vmin[i] <- min(res)/numFlips
  }
  list(v1 = v1, vrand = vrand, vmin = vmin)
}

## Evaluates the left-hand side of the Hoeffding inequality for a given distribution of the fraction of heads obtained
## during a series of coin tosses and a specified epsilon
hoeffding.lhs <- function(distribution, epsilon) {
  sum(abs(distribution - .5) > epsilon)/length(distribution)
}

## Evaluates the right-hand side of the Hoeffding inequality for a given sample size and epsilon
hoeffding.rhs <- function(N = 10, epsilon) {
  2*exp(-2*(epsilon^2)*N)
}

## Arguments: numTrials - Number of trials to pass to simulate.flips, numFlips - number of flips per trial, epsilon - vector of epsilon values on which to evaluate Hoeffding's inequality
## Return: List containing the means of the generated distributions of the fraction of heads obtained for the three respective coins
## Plots the left-hand side of the Hoeffding inequality against the right-hand side for the generated distributions of each of the three coins using the provided epsilon values
plot.hoeffding <- function(numTrials = 100000, numFlips = 10, epsilon = seq(0, 1, .1)) {
  simulation <- simulate.flips(numTrials, numFlips)
  probs <- lapply(simulation, function(x) sapply(epsilon, hoeffding.lhs, distribution = x))  # create a list of vectors of the values of the left-hand side of Hoeffding's inequality when evaluated for the distributions of each of the three coins across all of the provided epsilon values 
  layout(matrix(c(1, 2, 3), 1, 3))
  lapply(probs, function(x) {
    plot(epsilon, x, type = 'l', xlab = expression(epsilon), ylab = expression(P))
    lines(epsilon, hoeffding.rhs(numFlips, epsilon), col = 'red')
    legend('topright', c(expression("P[|" * nu - mu ~"|" > epsilon ~ "]"), expression(2*e^(-2*epsilon^2*N))), col = c('black', 'red'), lty = c(1,1), cex = 1.25, text.width = 0.35)
  })
  
  mtext("Probability plots for c_1, c_rand, and c_min, respectively", side = 3, line = -3, outer = TRUE)
  
  lapply(simulation, mean) # return v (in sample mean) for each of the coins
}

plot.hoeffding()  #Problems 1 & 2

