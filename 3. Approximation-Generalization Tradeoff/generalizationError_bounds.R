############### GENERALIZATION ERROR ###############
#
# There are various bounds for generalization error.  This script contains functions for calculating four such bounds
# given
#
# A function for computing the minimum sample size necessary to satisfy the VC bound for specified values of epsilon, 
# delta, and VC dimension is also provided.
#
# Each of these functions requires a growth function to be passed in as a parameter.  One growth function, which 
# uses N^(d_VC) as an approximation, is provided and used by default if no growth function is specified.  
#
############### DEFINITIONS ###############
#
# N = Sample Size
# d_VC = VC Dimension
# g = A hypothesis function for approximating a target function
# E_in(g) = In-Sample Error
# E_out(g) = Out-of-Sample Error
# m_H(N) = Growth Function:  Counts the most dichotomies on any N points; satisfies m_H(N) <= 2^N 
# epsilon = Omega = Generalization Error Tolerance
# eps = Actual Generalization Error
# delta = Upper bound on the probability that generalization error will be more than a specified value
# 1 - delta = Confidence that generalization error will be at most a specified value
#
# True Generalization Error: eps = E_out - E_in
#
# Generalization Error Tolerance: epsilon = Omega = sqrt((8/N)*ln(4*m_H(2N)/delta))
#
# VC Inequality: P[|E_in(g) - E_out(g)| > epsilon] <= 4*m_H(2N)*e^(-(1/8)*(epsilon^2)*N) = delta
#
############### RELEVANT BOUNDS ###############
#
# The following bounds all hold with probability at least 1 - delta:
#
# Rearranging the VC Inequality: E_out <= E_in + Omega 
#
# Original VC Bound: eps <= sqrt((8/N)*ln(4*m_H(2N)/delta))
#
# Rademacher Penalty Bound: eps <= sqrt((2*ln(2*N*m_H(N)))/N) + sqrt((2/N)*ln(1/delta)) + 1/N
#
# Parrondo and Van den Broek: eps <= sqrt((1/N)*(2*eps + ln((6*m_H(2*N))/delta)))
#
# Devroye: eps <= sqrt((1/(2*N))*(4*eps*(1 + eps) + ln((4*m_H(N^2))/delta)))

############### IMPLEMENTATION ###############

## A simple approximation for growth functions using N^(d_VC)
## If N <= d_VC, then the growth function can be computed exactly as 2^N because the sample can be shattered
## log.apply - When set to true, the natural log is applied to the approximation; useful for dealing with large numbers
m_H.approximate <- function(N, d_VC, log.apply = FALSE) {
  if(log.apply) {
    if(N <= d_VC)
      return(N*log(2))
    else
      return(d_VC*log(N))
  }
  
  if(N <= d_VC)
    return(2^N)
  
  else
    N^(d_VC)
}

## Calculates the minimum sample size necessary to satisfy the VC inequality; N >= (8/epsilon^2)ln((4*m_H(2*N))/delta)
## Uses an iterative method for solving N = f(N).  The method is as follows:
##    Choose a starting value N_0 and then iteratively set N_i+1 = f(N_i) 
##    Stop when N_i+1 approximates N_i within a specified margin.
##    Works when |f(N_i) - f(N_j)| <= c*|N_i - N_j| in the region where the iterations are taking place for some c < 1
## epsilon - Generalization error tolerance
## delta - Confidence level; there is (1 - delta)% confidence that the generalization error will be at most epsilon
## d_VC - VC dimension
## m_H - A growth function
## N_0 - Starting point for iterative approximation
## margin - Margin of error for iterative approximation
vcBound.computeSufficientN <- function(epsilon = 0.05, delta = 0.05, d_VC = 10, m_H = m_H.approximate, N_0 = 1, margin = 0.0001) {
  N_1 <- (8/epsilon^2)*log((4*m_H(2*N_0, d_VC))/delta)
  while(N_1 - N_0 > margin) {
    N_0 <- N_1
    N_1 <- (8/epsilon^2)*log((4*m_H(2*N_0, d_VC))/delta)
  }
  N_1
}

## Calculates the VC bound on generalization error for specified parameters
bound.vapnikChervonenkis <- function(N, delta = 0.05, d_VC = 50, m_H = m_H.approximate) {
  sqrt((8/N)*log((4*m_H(2*N, d_VC))/delta))
}

## Calculates the Rademacher Penalty bound on generalization error for specified parameters
bound.rademacherPenalty <- function(N, delta = 0.05, d_VC = 50, m_H = m_H.approximate) {
  sqrt((2*log(2*N*m_H(N, d_VC)))/N) + sqrt((2/N)*log(1/delta)) + 1/N
}

## Calculates the Parrondo and Van De Broek bound on generalization error for specified parameters
## The bound is an implicit bound on eps; thus, it is solved by iterative method
bound.parrondo_vanDenBroek <- function(N, delta = 0.05, d_VC = 50, m_H = m_H.approximate, eps_0 = 0, margin = 0.0001) {
  eps_1 <- sqrt((1/N)*(2*eps_0 + log((6*m_H(2*N, d_VC))/delta)))
  while(eps_1 - eps_0 > margin) {
    eps_0 <- eps_1
    eps_1 <- sqrt((1/N)*(2*eps_0 + log((6*m_H(2*N, d_VC))/delta)))
  }
  eps_1
}

## Calculates the Devroye bound on generalization error for specified parameters
## The bound is an implicit bound on eps; thus, it is solved by iterative method
## This bound is trivial and always satisfied for N < 3; iteration will not converge in this range
## log.eval - When set to true, the log expression in the bound will be expanded; useful for handling very large numbers
bound.devroye <- function(N, delta = 0.05, d_VC = 50, m_H = m_H.approximate, eps_0 = 0, margin = 0.0001, log.eval = TRUE) {
  if(log.eval)
    bound <- function(x) sqrt((1/(2*N))*(4*x*(1 + x) + log(4) + m_H(N = N^2, d_VC = d_VC, log.apply = TRUE) - log(delta)))
  else
    bound <- function(x) sqrt((1/(2*N))*(4*x*(1 + x) + log((4*m_H(N^2, d_VC))/delta)))
  eps_1 <- bound(eps_0)
  while(eps_1 - eps_0 > margin) {
    eps_0 <- eps_1
    eps_1 <- bound(eps_0)
  }
  
  eps_1
}

## Plots each of the four bounds across specified values of N
## Values of N < 3 will cause computation of the devroye bound to fail because of divergence in the iteration method
bounds.plot <- function(N = seq(3, 10003, 10), delta = 0.05, d_VC = 50, m_H = m_H.approximate, eps_0 = 0, margin = 0.0001, log.eval = TRUE) {
  vc <- numeric(length(N))
  rademacher <- numeric(length(N))
  parrondoVdB <- numeric(length(N))
  devroye <- numeric(length(N))
  for(i in 1:length(N)) {
    vc[i] <- bound.vapnikChervonenkis(N[i], delta, d_VC, m_H)
    rademacher[i] <- bound.rademacherPenalty(N[i], delta, d_VC, m_H)
    parrondoVdB[i] <- bound.parrondo_vanDenBroek(N[i], delta, d_VC, m_H, eps_0, margin)
    devroye[i] <- bound.devroye(N[i], delta, d_VC, m_H, eps_0, margin, log.eval)
  }
  par(mfrow=c(2,2))
  plot(N, vc, type = 'l', xlab = 'N', ylab = expression(epsilon), main = 'VC Bound')
  plot(N, rademacher, type = 'l', xlab = 'N', ylab = expression(epsilon), main = 'Rademacher Penalty Bound')
  plot(N, parrondoVdB, type = 'l', xlab = 'N', ylab = expression(epsilon), main = 'Parrondo & Van de Broek Bound')
  plot(N, devroye, type = 'l', xlab = 'N', ylab = expression(epsilon), main = 'Devroye Bound')
}

## Problem 1
vcBound.computeSufficientN() 

## Problem 2
bound.vapnikChervonenkis(10000)
bound.rademacherPenalty(10000)
bound.parrondo_vanDenBroek(10000)
bound.devroye(10000)

## Problem 3
bound.vapnikChervonenkis(5)
bound.rademacherPenalty(5)
bound.parrondo_vanDenBroek(5)
bound.devroye(5)

## Plots
bounds.plot() 