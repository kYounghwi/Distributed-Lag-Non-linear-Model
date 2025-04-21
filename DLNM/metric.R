# Define correlation metric 
Correlation <- function(y_true, y_pred){
  cor <- cor(y_true, y_pred)    # 실제치, 예측치와의 상관 계수
  return(cor)
}

# Define smape metric 
Smape <- function(actual, predicted){
  n <- length(actual)
  smape_val <- (1/n) * sum(2 * abs(actual - predicted) / (abs(actual) + abs(predicted)))
  return(smape_val)
}

# Define RMSE metric
Rmse <- function(actual, predicted) {
  val <- sqrt(mean((predicted - actual) ^ 2))
  return(val)
}

# Define RRSE metric
Rrse <- function(actual, predicted) {
  val <- sqrt(sum((predicted - actual) ^ 2) / sum((mean(actual) - actual) ^ 2))
  return(val)
}

# Define R2 metric
R_squared <- function(actual, predicted) {
  val <- 1 - (sum((predicted - actual) ^ 2) / sum((mean(actual) - actual) ^ 2))
  return(val)
}

# Define Mae metric
Mae <- function(actual, predicted) {
  val <- mean(abs(actual - predicted))
  return(val)
}