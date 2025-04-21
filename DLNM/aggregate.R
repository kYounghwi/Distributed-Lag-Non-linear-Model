
# 결과 집계
aggregate <- function(pred_lst){
  
  result_lst <- numeric(0)
  
  for(i in 1:23){
    
    result_lst <- c(result_lst, 0)
    
    for(j in 1:nrow(pred_lst)){
      
      result_lst[i] <- result_lst[i] + pred_lst[[j, i]]     # 해당 인덱스 모든 예측 값 더하기
      
    }
    result_lst[i] <- result_lst[i] / nrow(pred_lst)       # 나누기 (평균내기)
  }
  
  return(result_lst)
  
}