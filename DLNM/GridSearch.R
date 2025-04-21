

library(dlnm)
library(splines)
library(stringr)
library(lubridate)
library(dplyr)
library(mgcv)  


###################### Road Data ##########################

library(dlnm)

data <- read.csv("C:/Users/admin/Desktop/R/Shit.csv")
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

####################################################################################

# 가능한 df와 nk 값 설정

# SSEC_df <- c(4, 5, 6, 7, 8, 9, 10, 11)
# SSEC_nk <- c(3, 4, 5, 6)
# 
CRB_df <- c(4, 5, 6, 7, 8, 9, 10, 11)
CRB_nk <- c(3, 4, 5, 6)

미국_df <- c(4, 5, 6, 7, 8, 9, 10, 11)
미국_nk <- c(3, 4, 5, 6)

주식_df <- c(4, 5, 6, 7, 8, 9, 10, 11)
주식_nk <- c(3, 4, 5, 6)

ED_df <- c(4, 5, 6, 7, 8, 9, 10, 11)
#ED_degree <- c(1, 2, 3)
ED_nk <- c(3, 4, 5, 6)

# 상위 n개 결과, 매개 변수 저장될 객체

perform_val <- rep(Inf, 30)
co_val <- rep(Inf, 30)
sm_val <- rep(Inf, 30)

# SSEC_df_val <- rep(Inf, 30)
# SSEC_nk_val <- rep(Inf, 30)
# 
CRB_df_val <- rep(Inf, 30)
CRB_nk_val <- rep(Inf, 30)

미국_df_val <- rep(Inf, 30)
미국_nk_val <- rep(Inf, 30)

주식_df_val <- rep(Inf, 30)
주식_nk_val <- rep(Inf, 30)

ED_nk_val <- rep(Inf, 30)
ED_df_val <- rep(Inf, 30)
#ED_degree_val <- rep(Inf, 30)


# 값 업데이트 될 함수
update_val <- function(new_value, co, sm, CRB_df, CRB_nk, 미국_df, 미국_nk, 주식_df, 주식_nk, ED_df, ED_nk) {
  
  global <- get("perform_val", envir = .GlobalEnv)
  global_co <- get("co_val", envir = .GlobalEnv)
  global_sm <- get("sm_val", envir = .GlobalEnv)
  
  global_CRB_df <- get("CRB_df_val", envir = .GlobalEnv)
  global_CRB_nk <- get("CRB_nk_val", envir = .GlobalEnv)
  
  global_주식_df <- get("주식_df_val", envir = .GlobalEnv)
  global_주식_nk <- get("주식_nk_val", envir = .GlobalEnv)
  
  global_ED_df <- get("ED_df_val", envir = .GlobalEnv)
  global_ED_nk <- get("ED_nk_val", envir = .GlobalEnv)
  
  global_미국_df <- get("미국_df_val", envir = .GlobalEnv)
  global_미국_nk <- get("미국_nk_val", envir = .GlobalEnv)
  
  
  if (new_value < max(global)) {
    
    global_co[which.max(global)] <- co
    global_sm[which.max(global)] <- sm
    
    global_CRB_df[which.max(global)] <- CRB_df
    global_CRB_nk[which.max(global)] <- CRB_nk
    
    global_주식_df[which.max(global)] <- 주식_df
    global_주식_nk[which.max(global)] <- 주식_nk
    
    global_ED_df[which.max(global)] <- ED_df
    global_ED_nk[which.max(global)] <- ED_nk
    
    global_미국_df[which.max(global)] <- 미국_df
    global_미국_nk[which.max(global)] <- 미국_nk
    
    global[which.max(global)] <- new_value
  }
  
  assign("perform_val", global, envir = .GlobalEnv)
  assign("co_val", global_co, envir = .GlobalEnv)
  assign("sm_val", global_sm, envir = .GlobalEnv)
  
  assign("CRB_df_val", global_CRB_df, envir = .GlobalEnv)
  assign("CRB_nk_val", global_CRB_nk, envir = .GlobalEnv)
  
  assign("주식_df_val", global_주식_df, envir = .GlobalEnv)
  assign("주식_nk_val", global_주식_nk, envir = .GlobalEnv)
  
  assign("ED_df_val", global_ED_df, envir = .GlobalEnv)
  assign("ED_nk_val", global_ED_nk, envir = .GlobalEnv)
  
  assign("미국_df_val", global_미국_df, envir = .GlobalEnv)
  assign("미국_nk_val", global_미국_nk, envir = .GlobalEnv)
  
}

# Define correlation loss function 
Correlation <- function(y_true, y_pred){
  cor_value <- cor(y_true, y_pred)
  loss <- 1 - cor_value
  return(loss)
}

# Define smape loss function 
Smape <- function(actual, predicted){
  n <- length(actual)
  smape_val <- (1/n) * sum(2 * abs(actual - predicted) / (abs(actual) + abs(predicted)))
  return(smape_val)
}


# 모든 조합에 대해 반복
for(S_df in SSEC_df){
  for(S_nk in SSEC_nk){
    
    for(C_df in CRB_df){
      for(C_nk in CRB_nk){
            
            for(미_df in 미국_df){
              for(미_nk in 미국_nk){
                  ###################### Create Cross-basis ##########################
                  
                  
                  # SSEC     
                  varknots <- equalknots(train_data %>% pull(SSEC), df = 4)
                  lagknots <- logknots(x = 30, nk = 3)                            # x means maximum lag, nk is a number of knots or cut-offs
                  cb_SSEC <- crossbasis(train_data %>% pull(SSEC), lag = 30,
                                        argvar = list(fun = "ns", knots = varknots),
                                        arglag = list(knots = lagknots))
                  
                  # CRB     
                  varknots <- equalknots(train_data %>% pull(CRB), df = C_df)
                  lagknots <- logknots(x = 30, nk = C_nk)                            # x means maximum lag, nk is a number of knots or cut-offs
                  cb_CRB <- crossbasis(train_data %>% pull(CRB), lag = 30,
                                       argvar = list(fun = "ns", knots = varknots),
                                       arglag = list(knots = lagknots))
                  
                  # 미국 
                  varknots <- equalknots(train_data %>% pull(미국경제불확실성), df = 미_df)
                  lagknots <- logknots(x = 30, nk = 미_nk)                            # x means maximum lag, nk is a number of knots or cut-offs
                  cb_미국 <- crossbasis(train_data %>% pull(미국경제불확실성), lag = 30,
                                      argvar = list(fun = "ns", knots = varknots),
                                      arglag = list(knots = lagknots))
                  
                  
                  ###################### Create glm Model ##########################
                  
                  library(splines)
                  model <- glm(BDI_shifted ~ cb_SSEC + cb_CRB + cb_미국 + 
                                 ns(date, 52), family=gaussian(), data = train_data)    
                  
                  
                  ###################### Extract attributes of Cross-basis ##########################
                  
                  # to replicate the transformation (train set 모델에 적용 위해)
                                    
                  attrcb2 <- attributes(cb_SSEC)  
                  attrcb3 <- attributes(cb_CRB)  
                  attrcb4 <- attributes(cb_미국)        
                  
                  ###################### Apply new Cross-basis of Test set ##########################
                  
                  # 기존의 Cross-basis 기준에 맞는 Test set에 대한 새로운 Cross-basis 생성 후 모델 적용
                  
                  cb_SSEC <- do.call(crossbasis, list(x=test_data$SSEC, lag=attrcb2$lag, argvar=attrcb2$argvar,
                                                      arglag=attrcb2$arglag))
                  
                  cb_CRB <- do.call(crossbasis, list(x=test_data$CRB, lag=attrcb3$lag, argvar=attrcb3$argvar,
                                                     arglag=attrcb3$arglag))
                  
                  cb_미국 <- do.call(crossbasis, list(x=test_data$미국경제불확실성, lag=attrcb4$lag, argvar=attrcb4$argvar,
                                                    arglag=attrcb4$arglag))
                  
                  ###################### Prediction & Processing ##########################
                  
                  pred <- predict(model, test_data, type="response")
                  
                  plot(test_data$date, pred)
                  
                  pred <- na.omit(pred)
                  actual <- test_data$BDI_shifted
                  actual <- actual[-c(1:30)]
        
                  # 변환
                  pred <- exp(pred)
                  actual <- exp(actual)
                  
                  ###################### Plot ########################## 
                  
                  # plot(actual, col="blue", pch=19, ylim=c(min(actual, pred), max(actual, pred)), main=df, ylab=nk, xlab=degree)
                  #points(pred, col="red", pch=19)
                  
                  ###################### Performance ##########################
                  
                  smape <- Smape(pred, actual)
                  correlation <- Correlation(pred, actual)
                  
                  perform <- 0.7*smape + 0.3*correlation
                  
                  # ##
                  cat("Current Smape : ", smape, "\n")
                  cat("Current Correlation : ", correlation, "\n")
                  
                  
                  update_val(perform, correlation, smape, S_df, S_nk, C_df, C_nk, 미_df, 미_nk)
        
              }
            }
          }
        }
      }
    }



for(i in 1:30){
  
  cat("Perform : ", perform_val[i], " ")
  cat("Correlation : ", co_val[i], " ")
  cat("Smape : ", sm_val[i], "\n")
  
  # cat("SSEC_df: ", SSEC_df_val[i], " ")
  # cat("SSEC_nk: ", SSEC_nk_val[i], " ")
  
  cat("CRB_df: ", CRB_df_val[i], " ")
  cat("CRB_nk: ", CRB_nk_val[i], " ")
  
  cat("미국_df: ", 미국_df_val[i], " ")  
  cat("미국_nk: ", 미국_nk_val[i], " ")
  
  cat("주식_df: ", 주식_df_val[i], " ")
  cat("주식_nk: ", 주식_nk_val[i], " ")

  cat("ED_df: ", ED_df_val[i], " ")
  cat("ED_nk: ", ED_nk_val[i], " ")
  
  cat("\n")
  
}


