

**현재 BDI_shifted를 예측, 마지막 날짜로부터 23주의 데이터 예측할 수 있게 
30 지연기간을 포함해 53주를 Test Set으로 씀**
  
  #install.packages("dlnm")
  #install.packages("tidyverse")
  
  library(dlnm)
library(splines)
library(stringr)
library(lubridate)
library(dplyr)
library(mgcv)  


###################### Road Data ##########################


data <- read.csv("C:/Users/응쏘/Desktop/R/Shit.csv")
data$date <- as.Date(data$date, format="%Y-%m-%d")    # datetype 변경
head(data)
dim(data)

###################### Train & Test Data ##########################

last_date <- max(data$date)              # 마지막 날짜
six_months_ago <- last_date - 13*4*7     # 마지막 날짜로부터 13개월 전

# train
train_data <- subset(data, date < six_months_ago)    
dim(train_data)
colSums(is.na(train_data))  # NA 존재여부

# test
test_data <- subset(data, date >= six_months_ago)    
dim(test_data)
colSums(is.na(test_data))  # NA 존재여부
tail(test_data)

test_data$index

###################### Create Cross-basis ##########################


# SSEC     
varknots <- equalknots(train_data %>% pull(SSEC), df = 5)
lagknots <- logknots(x = 30, nk = 4)                            # x means maximum lag, nk is a number of knots or cut-offs
cb_SSEC <- crossbasis(train_data %>% pull(SSEC), lag = 30,
                      argvar = list(fun = "ns", knots = varknots),
                      arglag = list(knots = lagknots))

# CRB     
varknots <- equalknots(train_data %>% pull(CRB), df = 9)
lagknots <- logknots(x = 30, nk = 5)                            # x means maximum lag, nk is a number of knots or cut-offs
cb_CRB <- crossbasis(train_data %>% pull(CRB), lag = 30,
                     argvar = list(fun = "ns", knots = varknots),
                     arglag = list(knots = lagknots))

# Cape     
varknots <- equalknots(train_data %>% pull(Cape.Newbuilding.Prices), df = 9)
lagknots <- logknots(x = 30, nk = 6)                            # x means maximum lag, nk is a number of knots or cut-offs
cb_CAPE <- crossbasis(train_data %>% pull(Cape.Newbuilding.Prices), lag = 30,
                      argvar = list(fun = "ns", knots = varknots),
                      arglag = list(knots = lagknots))

# 미국 
varknots <- equalknots(train_data %>% pull(미국경제불확실성), df = 10)
lagknots <- logknots(x = 30, nk = 5)                            # x means maximum lag, nk is a number of knots or cut-offs
cb_미국 <- crossbasis(train_data %>% pull(미국경제불확실성), lag = 30,
                    argvar = list(fun = "ns", knots = varknots),
                    arglag = list(knots = lagknots))

# 주식  
varknots <- equalknots(train_data %>% pull(주식경제불확실성), fun = "bs", df = 4, degree = 3)
lagknots <- logknots(x = 30, nk = 5)                            # x means maximum lag, nk is a number of knots or cut-offs
cb_주식 <- crossbasis(train_data %>% pull(주식경제불확실성), lag = 30,
                    argvar = list(fun = "ns", knots = varknots),
                    arglag = list(knots = lagknots))

# EuroDoller 
varknots <- equalknots(train_data %>% pull(EuroDoller), df = 4)
lagknots <- logknots(x = 30, nk = 3)                            # x means maximum lag, nk is a number of knots or cut-offs
cb_ED <- crossbasis(train_data %>% pull(EuroDoller), lag = 30,
                    argvar = list(fun = "ns", knots = varknots),
                    arglag = list(knots = lagknots))



###################### Create glm Model ##########################

library(splines)
model <- glm(BDI_shifted ~ cb_SSEC + cb_CRB + cb_CAPE + cb_미국 + 
               ns(date, 52), family=gaussian(), data = train_data)   

###################### MSE & AIC ##########################

# 예측 성능 평가 (MSE)
fitted_ <- fitted(model)
actual_values <- train_data$BDI_shifted
actual_values <- actual_values[-c(1:30)]

length(fitted_)
length(actual_values)

# 지수화
fitted_ <- exp(fitted_)
actual_values <- exp(actual_values)

plot(actual_values, col="blue", type='l', lwd=2, ylim=c(min(actual_values, fitted_), max(actual_values, fitted_)), main='Model 1', ylab='BDI', xlab='Time')
lines(fitted_, col="red", lwd=2)
# 범례
legend("topright", legend=c("Prediction", "Actual"), col=c("red", "blue"), lwd=2)

# rmse
rmse <- sqrt(mean((fitted_ - actual_values)^2))
print(rmse)

# 더 확대한 결과

thresh <- round(length(fitted_)/4)

actual_values <- actual_values[-c(thresh:length(fitted_))]
fitted_ <- fitted_[-c(thresh:length(fitted_))]

plot(actual_values, col="blue", type='l', lwd=2, ylim=c(min(actual_values, fitted_), max(actual_values, fitted_)), main='Model 1', ylab='BDI', xlab='Time')
lines(fitted_, col="red", lwd=2)
# 범례
legend("topleft", legend=c("Prediction", "Actual"), col=c("red", "blue"), lwd=2)



###################### Extract attributes of Cross-basis ##########################

# BDI 이용 예측
pred_BDI <- crosspred(cb_BDI, model, cen = median(train_data$BDI), by = 0.1)

# BDI plot
plot(pred_BDI, xlab = "BDI", zlab = "BDI(log-converted)", theta = 200, phi = 40,
     lphi = 30, main = "3D graph of BDI effect")
plot(pred_BDI, "contour", xlab = "BDI(log-converted)", key.title = title("RR"),
     plot.title = title("Contour plot", xlab = "BDI", ylab = "Lag"))

diff_range <- apply(pred_BDI$matfit, 2, function(x) max(x) - min(x))
top_10 <- sort(diff_range, decreasing = TRUE)[1:10]       # 7 ~ 11
mean_top_10_BDI <- mean(top_10)
mean_top_10_BDI

# 특정 시점 기준
plot(pred_BDI, lag=9, ylab="Effect at 9 Lag", xlab="BDI")

#plot(pred_BDI, "slices", var = c(9, 8, 7), lag = c(10, 15, 20),
#      ci.level = 0.99, xlab = "BDI",
#      ci.arg = list(density = 20, col = grey(0.7)))

# 특정
#plot(pred_BDI, "slices", var = 7, ci = "n")
#for(i in 1:2) lines(pred_BDI, "slices", var = c(8, 9)[i], col = i + 2,
#                     lwd = 1.5)

# SSEC 이용 예측
pred_SSEC <- crosspred(cb_SSEC, model, cen = median(train_data$SSEC), by = 0.1)

# SSEC plot
plot(pred_SSEC, xlab = "SSEC", zlab = "BDI(log-converted)", theta = 200, phi = 40,
     lphi = 30, main = "3D graph of SSEC effect")
plot(pred_SSEC, "contour", xlab = "SSEC", key.title = title("RR"),
     plot.title = title("Contour plot", xlab = "SSEC", ylab = "Lag"))

diff_range <- apply(pred_SSEC$matfit, 2, function(x) max(x) - min(x))
top_10 <- sort(diff_range, decreasing = TRUE)[1:10]        # 7 ~ 11
mean_top_10_SSEC <- mean(top_10)
mean_top_10_SSEC

# 특정 시점 기준
plot(pred_SSEC, lag=8, ylab="Effect at 8 Lag", xlab="SSEC")

# CRB 이용 예측
pred_CRB <- crosspred(cb_CRB, model, cen = median(train_data$CRB), by = 0.1)

# CRB plot
plot(pred_CRB, xlab = "CRB", zlab = "BDI(log-converted)", theta = 200, phi = 40,
     lphi = 30, main = "3D graph of CRB effect")
plot(pred_CRB, "contour", xlab = "CRB", key.title = title("RR"),
     plot.title = title("Contour plot", xlab = "CRB", ylab = "Lag"))

diff_range <- apply(pred_CRB$matfit, 2, function(x) max(x) - min(x))
top_10 <- sort(diff_range, decreasing = TRUE)[1:10]         # 9 ~13
mean_top_10_CRB <- mean(top_10)
mean_top_10_CRB

# 특정 시점 기준
plot(pred_CRB, lag=11, ylab="Effect at 11 Lag", xlab="CRB")

# 미국경제불확실성 이용 예측
pred_미국 <- crosspred(cb_미국, model, cen = median(train_data$미국경제불확실성), by = 0.1)

# 미국경제불확실성 plot
plot(pred_미국, xlab = "미국경제불확실성", zlab = "BDI(log-converted)", theta = 200, phi = 40,
     lphi = 30, main = "3D graph of 미국경제불확실성 effect")
plot(pred_미국, "contour", xlab = "미국경제불확실성", key.title = title("RR"),
     plot.title = title("Contour plot", xlab = "미국경제불확실성", ylab = "Lag"))

diff_range <- apply(pred_미국$matfit, 2, function(x) max(x) - min(x))
top_10 <- sort(diff_range, decreasing = TRUE)[1:10]     # 14 ~ 18
mean_top_10_미국 <- mean(top_10)
mean_top_10_미국

# 특정 시점 기준
plot(pred_미국, lag=16, ylab="Effect at 16 Lag", xlab="미국")

# 주식경제불확실성 이용 예측
pred_주식 <- crosspred(cb_주식, model, cen = median(train_data$주식경제불확실성), by = 0.1)

# 주식경제불확실성 plot
plot(pred_주식, xlab = "주식경제불확실성", zlab = "RR", theta = 200, phi = 40,
     lphi = 30, main = "3D graph of 주식경제불확실성 effect")
plot(pred_주식, "contour", xlab = "주식경제불확실성", key.title = title("RR"),
     plot.title = title("Contour plot", xlab = "주식경제불확실성", ylab = "Lag"))

diff_range <- apply(pred_주식$matfit, 2, function(x) max(x) - min(x))
top_10 <- sort(diff_range, decreasing = TRUE)[1:10]
mean_top_10_주식 <- mean(top_10)
mean_top_10_주식


# EuroDoller 이용 예측
pred_ED <- crosspred(cb_ED, model, cen = median(train_data$EuroDoller), by = 0.1)

# EuroDoller plot
plot(pred_ED, xlab = "EuroDoller", zlab = "BDI(log-converted)", theta = 200, phi = 40,
     lphi = 30, main = "3D graph of EuroDoller effect")
plot(pred_ED, "contour", xlab = "EuroDoller", key.title = title("RR"),
     plot.title = title("Contour plot", xlab = "EuroDoller", ylab = "Lag"))

diff_range <- apply(pred_ED$matfit, 2, function(x) max(x) - min(x))
top_10 <- sort(diff_range, decreasing = TRUE)[1:10]        # 7 ~ 11
mean_top_10_ED <- mean(top_10)
mean_top_10_ED

# 특정 시점 기준
plot(pred_ED, lag=8, ylab="Effect at 8 Lag", xlab="ED")

df <- data.frame(floats = c(mean_top_10_BDI, mean_top_10_SSEC, mean_top_10_CRB, mean_top_10_미국, mean_top_10_ED)
                 , strings = c("BDI", "SSEC", "CRB", "미국", "ED"))

# sort by the float column in ascending order
df_sorted <- df[order(-df$floats), ]
df_sorted

###################### Extract attributes of Cross-basis ##########################

# to replicate the transformation (train set 모델에 적용 위해)

attrcb2 <- attributes(cb_SSEC)  
attrcb3 <- attributes(cb_CRB)   
attrcb4 <- attributes(cb_CAPE)
attrcb5 <- attributes(cb_미국)


###################### Apply new Cross-basis of Test set ##########################

# 기존의 Cross-basis 기준에 맞는 Test set에 대한 새로운 Cross-basis 생성 후 모델 적용

cb_SSEC <- do.call(crossbasis, list(x=test_data$SSEC, lag=attrcb2$lag, argvar=attrcb2$argvar,
                                    arglag=attrcb2$arglag))

cb_CRB <- do.call(crossbasis, list(x=test_data$CRB, lag=attrcb3$lag, argvar=attrcb3$argvar,
                                   arglag=attrcb3$arglag))

cb_CAPE <- do.call(crossbasis, list(x=test_data$Cape.Newbuilding.Prices, lag=attrcb4$lag, argvar=attrcb4$argvar,
                                    arglag=attrcb4$arglag))

cb_미국 <- do.call(crossbasis, list(x=test_data$미국경제불확실성, lag=attrcb5$lag, argvar=attrcb5$argvar,
                                    arglag=attrcb5$arglag))


###################### Prediction & plot ##########################

pred <- predict(model, test_data, type="response")

plot(test_data$date, pred)     #  각자 plot
plot(test_data$date, test_data$BDI_shifted)

# Define correlation metric 
Correlation_loss <- function(y_true, y_pred){
  cor_value <- cor(y_true, y_pred)    # 실제치, 예측치와의 상관 계수
  loss <- 1 - cor_value               # 0 ~ 2 사이 값 만들기
  return(loss)
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

# 변환 전

pred <- na.omit(pred)
actual <- test_data$BDI_shifted
actual <- actual[-c(1:30)]

#plot(actual, col="blue", pch=19, ylim=c(min(actual, pred), max(actual, pred)))
#points(pred, col="red", pch=19)

#smape <- Smape(pred, actual)
#correlation_loss <- Correlation_loss(pred, actual)

#cat("Smape : ", smape, "\n")
#cat("Correlation Loss : ", correlation_loss)


date <- test_data$date
typeof(date)
date <- date[-c(1:30)]
date <- as.Date(date)

# 변환 후

pred <- exp(pred)
actual <- exp(actual)

plot(date, actual, col="blue", type='l', lwd=2, ylim=c(min(actual, pred), max(actual, pred)), main='Model 1', ylab='BDI', xlab='Time')
lines(date, pred, col="red", lwd=2)

# 범례
legend("topright", legend=c("Prediction", "Actual"), col=c("red", "blue"), lwd=2)

smape <- Smape(pred, actual)
correlation <- Correlation_loss(pred, actual)
rmse <- Rmse(pred, actual)
rrse <- Rrse(pred, actual)
r2 <- R_squared(pred, actual)
mae <- Mae(pred, actual)

cat("Smape : ", smape, "\n")
cat("RMSE : ", rmse, '\n')
cat("Mae : ", mae, '\n')
cat("RRSE : ", rrse, '\n')
cat("R2 : ", r2, '\n')
cat("Correlation : ", correlation, '\n')




