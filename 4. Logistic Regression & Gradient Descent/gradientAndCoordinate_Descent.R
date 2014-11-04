

gradient.descent <- function(expr, parameters, values, goal = quote(eval(expr <= -Inf)), maxIter = Inf, eta = 0.1) {
  for(i in 1:length(values)) {
    assign(parameters[i], values[[i]])
  }
  iterations <- 0
  repeat {
    d_values <- -eta*attributes(eval(deriv(expr, parameters)))[[1]]
    values <- values + d_values
    for(i in 1:length(values)) {
      assign(parameters[i], values[[i]])
    }
    iterations <- iterations + 1
    if(eval(goal) || iterations >= maxIter)
      break
  }
  list(parameter_values = values, f = eval(expr), numIterations = iterations)
}


coordinate.descent <- function(expr, parameters, values, goal = quote(eval(expr <= -Inf)), maxIter = Inf, eta = 0.1) {
  for(i in 1:length(values)) {
    assign(parameters[i], values[i])
  }
  iterations <- 0
  repeat {
    for(i in 1:length(values)) {
      d_value <- -eta*attributes(eval(deriv(expr, parameters[i])))[[1]]
      values[i] <- values[i] + d_value
      assign(parameters[i], values[i])
    }
    iterations <- iterations + 1
    if(eval(goal) || iterations >= maxIter)
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
