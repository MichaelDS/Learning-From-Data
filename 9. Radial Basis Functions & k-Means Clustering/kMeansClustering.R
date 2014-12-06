## Generates the input data and uses the target function f(x) = sign(x2 - x1 + 0.25*sin(pi*x1)) to generate response values
## Ensures that the response values do not all belong to a single class
data.generate <- function(N = 100, limits = c(-1, 1)) { 
  repeat{
    # generate the points
    x1 <- runif(N, limits[1], limits[2])
    x2 <- runif(N, limits[1], limits[2])
    
    # compute responses
    y <- sign(x2 - x1 + 0.25*sin(pi*x1))
    
    if(abs(sum(y)) != length(y))  # make sure the points are not all of the same class
      break
  } 
  data.frame(x1, x2, y) # put the training set together and return it
}

## Can also use R's dist() function
dist.euclidian <- function(x1, x2) {
  sqrt(sum((as.vector(x1) - as.vector(x2))^2))  # as.vector() avoids potentially having to deal with dimension conformability
}

## X - A data frame of points to cluster
## Returns a matrix containing the coordinates of the cluster centers and a vector of cluster indices corresponding to the input data
cluster.kMeans <- function(X, nCenters = 9, clusters.plot = FALSE) {
  # initialize cluster centers using random sample of points
  # centers do not have to be equal to any points; this is an artifact of initialization.  The centers will shift as the algorithm iterates
  centers <- X[sample(1:nrow(X), nCenters), ]  
  X$cluster <- 0
  
  f <- function(i, j) dist.euclidian(X[i, -ncol(X)], centers[j,])  # define function to be passed to outer
  
  repeat {
    X$cluster <- apply(outer(1:nrow(X), 1:nrow(centers), Vectorize(f)), 1, function(x) which(x == min(x)))  # cluster each point according to the nearest center; use outer() to generate a points vs centers distance matrix, then choose the nearest center using apply() and which()
    if(0 %in% sapply(1:nCenters, function(x) length(which(X$cluster == x)))) {  # check if there are any empty clusters
      centers <- X[sample(1:nrow(X), nCenters), -ncol(X)]                       # reset centers
      X$cluster <- 0                                                            # reset cluster assignments
      next                                                                      # restart the algorithm
    }
    newCenters <- matrix(0, nrow(centers), ncol(centers), dimnames = list(c(), c('x1', 'x2')))
    for(i in 1:nrow(centers)) {                                                 # recalculate centers
      newCenters[i, ] <- colMeans(X[which(X$cluster == i), -ncol(X)])
    }
    if(identical(centers, newCenters))  # centers becomes a matrix with no labels (as opposed to a data frame) after first iteration, so identical can correctly compare it with newCenters
      break
    centers <- newCenters
  }
  
  if(clusters.plot) {
    library(ggplot2)
    library(ggthemes)
    centersDF <- data.frame(x1 = centers[, 'x1'], x2 = centers[, 'x2'], cluster = seq(1:nrow(centers)))
    p <- ggplot(data = X, aes(x1, x2, colour = as.factor(cluster))) + geom_point() +
      scale_colour_tableau() + theme_solarized(light = 'false') +
      labs(colour = 'cluster') + ggtitle('Clustered Data') +
      geom_point(data = centersDF, aes(colour = as.factor(cluster)), shape = 8, size = 2.5) +
      guides(colour = guide_legend(override.aes = list(size=2, shape = 16)))
    #scale_color_brewer(palette = 'Pastel1')   # decided to go with Tableau colors and dark solarized theme instead of a brewer set
    print(p)
  }
  
  list(centers = centers, clusters = X$cluster)
}

## Example call to cluster.kMeans() using the plotting feature
data <- data.generate()
cluster.kMeans(data[, -length(data)], nCenters = 9, clusters.plot = TRUE)
