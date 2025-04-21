
#%%

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_rrse(actual, predicted):
    # 실제 값과 예측 값의 차이 계산
    residuals = actual - predicted

    # 제곱 오차 계산
    squared_residuals = residuals ** 2

    # 평균 제곱 오차 계산
    mse = np.mean(squared_residuals)

    # 평균 제곱 오차의 제곱근을 계산
    rmse = np.sqrt(mse)

    # 실제 값의 평균 계산
    mean_actual = np.mean(actual)

    # RRSE 계산
    rrse = rmse / mean_actual

    return rrse

def calculate_smape(actual, forecast):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters:
    - actual: 실제 값의 배열 또는 리스트
    - forecast: 예측 값의 배열 또는 리스트

    Returns:
    - smape: SMAPE 값
    """

    # 0으로 나누기를 피하기 위해 분모에 작은 값 추가
    denominator = np.abs(actual) + np.abs(forecast) + 1e-9

    # SMAPE 계산
    smape = np.mean(2 * np.abs(forecast - actual) / denominator)

    return smape


#%%

df = pd.read_csv('C:/Users/User/Desktop/ExpFile/Exp_DLNM/Data/BDISet.csv', index_col=0)
df = df['BDI']
df.index = pd.to_datetime(df.index)

print(df)

#%%

p_lst = [0, 1, 2, 3]
d_lst = [0, 1, 2, 3]
q_lst = [0, 1, 2, 3]

test_size = 23  

train = df[:-test_size]
test = df[-test_size:]

#%%

best = -1 * 10**9

for p in p_lst:
    for d in d_lst:
        for q in q_lst:
            
            model = ARIMA(train, order=(p, d, q))
            results = model.fit()
            
            forecast_steps = test_size  # 예측할 스텝 수, 적절한 값으로 조절이 필요합니다.
            forecast = results.get_forecast(steps=forecast_steps)
            
            prediction = np.array(forecast.predicted_mean)
            actual = np.array(test)
            
            rmse = np.sqrt(mean_squared_error(prediction, actual))
            r2 = r2_score(prediction, actual)
            
            if best < r2:
                best = r2
            
            print(f'P: {p}, D: {d}, Q: {q} - R2: {r2:.4f}, RMSE: {rmse:.4f}')
            
            plt.plot(actual, label='actual')
            plt.plot(prediction, label = 'prediction')
            plt.legend()
            plt.show()
            

#%%

################################################## Test ##################################################

#%%

df = pd.read_csv('C:/Users/User/Desktop/ExpFile/Exp_DLNM/Data/BDISet.csv', index_col=0)
df = df['BDI']
df.index = pd.to_datetime(df.index)

test_size = 23

train = df[:-test_size]
test = df[-test_size:]

#%%

# p, d, q는 ARIMA의 파라미터로 각각 자동회귀(AR), 차분(Difference), 이동평균(MA)을 나타냅니다.
p, d, q = 2, 3, 2

model = ARIMA(train, order=(p, d, q))
results = model.fit()

forecast_steps = test_size  # 예측할 스텝 수, 적절한 값으로 조절이 필요합니다.
forecast = results.get_forecast(steps=forecast_steps)

#%%

prediction = np.array(forecast.predicted_mean)
actual = np.array(test)

plt.plot(actual, label='actual')
plt.plot(prediction, label = 'prediction')
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(prediction, actual))
mae = mean_absolute_error(prediction, actual)
r2 = r2_score(prediction, actual)
rrse = calculate_rrse(actual, prediction)
smape = calculate_smape(actual, prediction)

print(f'SMAPE: {smape:.4f} / RMSE: {rmse:.4f} / MAE: {mae:.4f} / R2: {r2:.4f} / RRSE: {rrse:.4f}')
print(prediction)

#%%

























