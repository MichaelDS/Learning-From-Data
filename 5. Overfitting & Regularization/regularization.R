############### OVERFITTING ###############
# 
# Overfitting occurs when a model describes random error or noise instead of the underlying relationship.  It generally
# occurs when a model is excessively complex, such as having too many parameters relative to the number of observations.  
# A model that has been overfit will generally have poor predictive performance, as it is fitting a 'pattern' that does 
# not actually exist.  Take, for example, 5 data points from a noisy quadratic target function.  A 4th-order polynomial 
# will be able to fit all of the points perfectly, reducing E_in to 0; however, E_out will most likely be extremely 
# large.  On the other hand, a simpler quadratic model may not fit the points perfectly, but its approximation of the 
# target function will be much better, thus producing a much smaller E_out.
# 
# Overfitting is conceptually different from bad generalization in that generalization worsens as the model becomes more
# complex (more effective degrees of freedom or VC dimension), even if the out-of-sample performance is improving, 
# whereas overfitting happens when out-of-sample performance worsens even though in-sample performance continues to 
# improve with increased model complexity.
# 
# Overfitting happens because of noise; a model that is too complex will interpret noise as a pattern and fit accordingly,
# resulting in poor out-of-sample performance.  One can say that noise causes the model to 'hallucinate' a pattern.
# 
# Noise can be though of as falling into two categories; stochastic noise and deterministic noise.  Stochastic noise
# is the manifestation of the randomness inherent to the distribution of the output with respect to the input.  
# The source of deterministic noise is the model's inability to perfectly replicate the target function (bias); 
# when the model tries to capture the higher complexity of a more complex target function, it extrapolates a false 
# pattern.  The main differences between stochastic noise and deterministic noise are that deterministic noise depends 
# on the hypothesis set (more complex hypothesis sets will result in lower deterministic noise, albeit more model 
# variance and a higher potential for fitting stochastic noise), and deterministic noise is fixed for a given x.  
# 
# Both types of noise can be treated the same as they each represent things about the data that can not be captured.
# Stochastic noise is pattern-less, there is nothing to capture, and deterministic noise is beyond the model's ability
# to capture.  The effects of overfitting are mitigated as the size of the training set increases.
#
# Bias-Variance decomposition in the presence of stochastic noise:
# E_(D,eps)[(g^D(x) - f(x))^2] = E_(D,x)[(g^D(x) - g_bar(X))^2] + E_x[(g_bar(x) - f(x))^2] + E_(eps,x)[eps(x)^2] 
#                              = var(x) + bias(x) + eps(x)^2
#                              = variance + bias(deterministic noise) + stochastic_noise
#
############### REGULARIZATION ###############
#
# Regularization refers to a process of introducing additional information in order to solve an ill-posed problem or to
# prevent overfitting.  This information is usually of the form of a penalty for complexity, such as restrictions for 
# smoothness or bounds on the vector space norm.  A theoretical justification for regularization is that it attempts to 
# impose Occam's razor on the solution.  From a Bayesian point of view, many regularization techniques correspond to 
# imposing certain prior distributions on model parameters.
#
# There are two approaches to regularization; the mathematical approach seeks to solve ill-posed problems in function 
# approximation by imposing smoothness constraints whereas the heuristic approaches handicap the minimization of E_in.  
# The best utility for the mathematical approach in practical machine learning is to develop the mathematics in a 
# specific case and then interpret the mathematical result in such a way to get an intuition that will apply when the
# assumptions do not apply (almost always in practical problems).  This intuition can then be used to guide heuristic
# methods of regularization.
#
# Regularization generally reduces the model variance significantly at the expense of slightly increasing the bias. 
# It handicaps the fit on both the noise and the signal, with the handicap on the noise being more significant.  
# Regularization allows us to fill in the gap between models of differing complexity and potentially discover an 
# in-between solution that optimizes the fit.
#
# One popular for of regularization for polynomial fitting is known as weight decay.  It imposes a soft constraint on 
# the weights on the linear model as follows:
#
# SIGMA(q = 0, Q) (w_q)^2 = SIGMA(q = 0, Q) t(w)%*%w <= C
#
# This is a proper subset of the full model; thus, intuition suggests that this constraint will result in better
# generalization.  The error measure being minimized under this constraint can be mathematically shown to be the 
# following:
#
# In order to minimize E_in(w) = (1/N)*t((Z%*%w - y))%*%(Z%*%w - y), subject to t(w)%*%w <= C
# minimize E_in(w) + (lambda/N)*t(w)%*%w, where lambda is a constant of proportionality that corresponds inversely
#                                         with the budget constant, C.
# This can be done in one step, as follows:
# 
# w_reg = (t(Z)%*%Z + lambda*I)^-1 %*% t(Z)%*%y
# 
# Notice that this formula makes follows intuition. For very large lambda, this reduces to approximately 
# (1/lambda)*t(Z)%*%y, where lambda is huge; this knocks down the magnitude of the weights.  Lambda is chosen in a 
# principled way, via validation, in order to optimize the model's out-of-sample performance.  Choosing too small of a
# lambda would not minimize overfitting, whereas choosing a lambda that is too large will result in underfitting.  
# Although the choice of regularizer is heuristic, choosing lambda in a principled way will indicate when a poor 
# regularizer has been chosen because lambda will simply be 0.  
#
# This method is called weight decay because the weights are decreased from one iteration to the next.  It applies to
# neural networks, where the weights "decay" in each step of gradient descent at a rate proportional to lambda.
#
# There are other variations of weight decay.  For example, emphasis can be placed on certain weights:
#
# SIGMA(q = 0, Q) gamma_q * (w_q)^2 <= C, where gamma_q expresses an emphasis
# For example, if gamma_q = 2^q, then the regularizer is placing a huge emphasis on higher-order terms; thus, it is
# trying to find a low-order fit, as high order terms will use up the budget quickly.
# On the other hand, gamma_q = 2^-q tries to find a high-order fit.
#
# In neural networks, this form is used to place emphasis on different weights at different layers.  This has been 
# found to be the best way to apply weight decay to neural networks.  The regularizer used to do this is as follows:
#
# Tikhonov regularizer: t(w)%*%t(GAMMA)%*%GAMMA%*%w
#
# This regularizer is a general quadratic form capable of capturing the effects of various different regularizers
# given the proper choice of GAMMA.  Weight decay in neural networks has a correspondence with the simplicity of the
# function being implemented in terms of the size of the weights.  Small weights correspond to a simple linear model
# whereas larger weights approach a logical dependency.  Another regularizer for for neural networks is known as 
# soft weight elimination.  Instead of combinatorially figuring out which weights to keep and which to eliminate
# (which would be extremely difficult in terms of optimization), small weights are de-emphasized and large weights
# are left mostly intact by via the following regularizer:
#
# OMEGA(w) = SIGMA(i,j,l) (w_(ij)^l)^2/(B^2 + (w_(ij)^l)^2)
#
# For small w, B dominates the denominator and the regularizer will end up diminishing the weights, whereas for large
# w, the regularizer approaches 1.
#
# Practical rule:  
# Stochastic noise is 'high-frequency'
# Deterministic noise is also non-smooth
# ==> Constrain learning toward smoother hypotheses (by convention, smaller weights are generally smoother)
# This will help prevent the model from fitting noise
#
# General form of augmented error; calling the regularizer OMEGA = OMEGA(h), we minimize:
#
# E_aug(h) = E_in(h) + (lambda/N)*OMEGA(h)
#
# Recall from VC analysis that: 
# 
# E_out(h) <= E_in(h) + OMEGA(H), (OMEGA is generalization error here)
#
# E_aug is better than E_in as a proxy for E_out.
#
# Guiding principle for choosing a regularizer:
# The regularizer should constrain in the direction of 'smoother' or simpler
#
# Early stopping is another form of regularization that is done through the optimizer which stops optimization of 
# E_in before overfitting occurs.  When to stop done is determined in a principled way via validation.
#
############### DEFINITIONS ###############
#
# lambda = A constant whose value determines how much emphasis is placed on the regularization term during fitting
# N = Sample Size
# f = The target function
# h, g = A hypothesis function for approximating a target function
# h^D, g^D = A hypothesis function for approximating a target function given a training set of size N, D
# E_D = Expectation with respect to all possible training sets of size N, D
# E_x = Expectation with respect to x
# E_in(h) = In-Sample Error
# E_out(h) = Out-of-Sample Error
# eps(x)^2 = Energy of stochastic noise
#
############### EXPERIMENT ###############
#
# This experiment implements classification via linear regression with a non-linear transformation and explores the 
# effects of using weight decay to regularize it using various values of lambda
#
# The files in.txt and out.txt contain a training set and test set, respectively.  Each line of the files corresponds
# to a two-dimensional input x = (x1, x2), so that X = R^2, followed by the corresponding label from Y = {-1, 1}.  
# Linear regression is applied with the following non-linear transformation:
#
# phi(x1, x2) = (1, x1, x2, x1^2, x2^2, x1*x2, |x1 - x2|, |x1 + x2|)
#
# Linear regression is then applied using the same transformation and weight decay; that is, the term 
# (lambda/N)*SIGMA(i = 0, 7) (w_i)^2 is added to the squared in-sample error.  The experiment is repeated for values
# of lambda = {10^-3, 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3}.  Among these values, lambda = 10^-1 appears to 
# minimize overfitting and produce the smallest out-of-sample error.  Values of lambda >= 10^0 appears to cause 
# under-fitting, as they produce greater out-of-sample error than is achieved without regularization.  Values of 
# lambda <= 10^-2 perform comparably to using the model without regularization.

### OPTIONAL TODO
# Plot classified y against x1 and x2; plot the decision boundary and observe how it changes for given lambda
# Look into filled contours in ggplot2

############### IMPLEMENTATION ###############

## Performs the non-linear transformation phi(x1, x2) = (1, x1, x2, x1^2, x2^2, x1*x2, |x1 - x2|, |x1 + x2|)
## Returns the transformed input 
## D- Complete training set with two-dimensional input
transform.phi <- function(D) {
  x1 <- D[1]
  x2 <- D[2]
  phi <- cbind(1, x1, x2, x1^2, x2^2, x1*x2, abs(x1 - x2), abs(x1 + x2))
  phi
}

## Performs classification using linear regression and, if specified, uses a non-linear transformation and regularization
## Returns measures of the in-sample and out-of-sample errors achieved
## D_train - Training set
## D_test - Test set
## transform - A function which returns a non-linear transformation on the input of the training set
## lambda - Strength of regularization
regression.classify <- function(D_train, D_test, transform = NULL, lambda = 0) {
  y_train <- D_train[[length(D_train)]] 
  y_test <- D_test[[length(D_test)]]
  if(is.null(transform)) {               # if no transformation is specified, simply use the provided input
    X <- as.matrix(cbind(1, D_train[1:length(D_train)-1]))
    X_test <- as.matrix(cbind(1, D_test[1:length(D_test)-1]))
  }
  else {                                 # otherwise, perform the non-linear transformation
    X <- as.matrix(transform(D_train))
    X_test <- as.matrix(transform(D_test))
  }

  w <- solve(t(X)%*%X + lambda*diag(nrow(t(X)%*%X)))%*%t(X)%*%y_train  # one-step learning with regularization

  # Apply the final hypothesis to the inputs and calculate E_in and E_out
  y_model <- sign(t(w)%*%t(X))
  E_in <- sum(y_model != y_train)/length(y_model)
  
  y_model <- sign(t(w)%*%t(X_test))
  E_out <- sum(y_model != y_test)/length(y_model)
  
  list(E_in = E_in, E_out = E_out) # return E_in and E_out
}

## Read data sets for problems 2-6
train <- read.table('in.txt')
test <- read.table('out.txt')

regression.classify(train, test, transform.phi) # Problem 2

regression.classify(train, test, transform.phi, lambda = 10^-3) # Problem 3

regression.classify(train, test, transform.phi, lambda = 10^3) # Problem 4

## Problems 5 & 6
regression.classify(train, test, transform.phi, lambda = 10^2)
regression.classify(train, test, transform.phi, lambda = 10^1)
regression.classify(train, test, transform.phi, lambda = 10^0)
regression.classify(train, test, transform.phi, lambda = 10^-1)
regression.classify(train, test, transform.phi, lambda = 10^-2)

