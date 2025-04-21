#%%
import os
import warnings
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.data.encoders import EncoderNormalizer

torch.set_float32_matmul_precision('high')

#%%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')
print(torch.__version__)

#%%

import tensorflow as tf
import tensorboard as tb

#%%

################################# Data Load #################################

model_sample = 3

data = pd.read_csv('C:/Users/User/Desktop/Expfile/Exp_DLNM/Data/Original.csv', index_col=0)
column_to_remove = ['석탄', '철광석', '주식경제불확실성', 'EuroDoller', 'NASDAQ',
                    'Panamax Newbuilding Prices', 'BDI', 'USD/JPY']
data = data.drop(column_to_remove, axis=1)

n_features = data.shape[1]
target_column = n_features - 1

# 중복된 인덱스 확인
duplicated_index = data.index.duplicated()
# 중복된 index에 해당하는 행 찾기
duplicated_rows = data[duplicated_index]

# 중복된 index에 해당하는 행 제거
data = data[~duplicated_index]

################################# Data Preprocessing #################################

target_name = data.columns[target_column]
features_name = [data.columns[i] for i in range(n_features)]

# add time index
data.index = pd.to_datetime(data.index)
data['Date'] = data.index
# data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
# data["time_idx"] -= data["time_idx"].min()

data['TimeSeries'] = data.reset_index().index + 1

# add additional features
data["year"] = data.Date.dt.year.astype(str).astype("category")  # categories have be strings
data["month"] = data.Date.dt.month.astype(str).astype("category")  # categories have be strings
data["day"] = data.Date.dt.day.astype(str).astype("category")  # categories have be strings
data['weekday'] = data.Date.dt.weekday.astype(str).astype("category")


max_prediction_length = 12
max_encoder_length = 120

training_cutoff = data["TimeSeries"].max() - max_prediction_length
data["constant_group_id"] = "constant_value"
data["constant_group_id"] = data["constant_group_id"].astype("category")
        
# ========================== EncoderNormalizer ========================== #

encoder_normalizer = EncoderNormalizer(
    method="robust",
    method_kwargs={
        "center": 0.5,
        "lower": 0.25,
        "upper": 0.75,
    }
)

# ========================== Split & Preprocessing ========================== #


train = data.iloc[:800]
valid = data.iloc[800:]

print(f'Shape / Train Data: {train.shape}, Valid Data: {valid.shape}')

training = TimeSeriesDataSet(
    data[lambda x: x.TimeSeries <= training_cutoff],
    time_idx="TimeSeries",
    target=target_name,
    #group_ids: 데이터를 그룹화하는 데 사용되는 ID
    group_ids = ['constant_group_id'],
    min_encoder_length=max_encoder_length,# // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    
    #static_categoricals: 시간에 따라 변하지 않는 범주형 데이터
    static_categoricals=['constant_group_id'],
    # static_reals: 시간에 따라 변하지 않는 실수형 데이터
    static_reals=[],
    # variable_groups: 변수 그룹을 나타내는 데이터
    variable_groups={},
    # time_varying_known_categoricals: 시간에 따라 변하고 이미 알려진 범주형 데이터
    time_varying_known_categoricals=['month'],
    # time_varying_known_reals: 시간에 따라 변하고 이미 알려진 실수형 데이터
    time_varying_known_reals=[],
    # time_varying_unknown_categoricals: 시간에 따라 변하지만 알려지지 않은 범주형 데이터
    time_varying_unknown_categoricals=[],
    # time_varying_unknown_reals: 시간에 따라 변하지만 알려지지 않은 실수형 데이터
    time_varying_unknown_reals=features_name,
    target_normalizer=encoder_normalizer,
    # 불규칙한 데이터셋 허용
    # allow_missing_timesteps=True,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# predict=True를 설정하여 각 시계열 데이터의 마지막 max_prediction_length 시점을 예측하도록 합니다.
# stop_randomization=True를 설정하여 검증 데이터셋을 생성하는 동안 시계열의 무작위 샘플링을 중지합니다. 
train_data = TimeSeriesDataSet.from_dataset(training, train, predict=False, stop_randomization=True)
valid_data = TimeSeriesDataSet.from_dataset(training, valid, predict=False, stop_randomization=True)

batch_size = 64

train_dataloader = train_data.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = valid_data.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

print(len(train_dataloader), len(val_dataloader))


hidden_lst = [64, 128, 256, 512]

for h in hidden_lst:
    
    # Each Model sample
    for m_s in range(model_sample):
        
        save_name = f"TFT_{h}hidden_{m_s}Modelsample"
        print(save_name)
        
        ################################# Train Setting #################################
        
        Epoch = 500
        earlyStopPatience = 100
        lr = 0.0001
    
        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=earlyStopPatience, verbose=True, mode="min")
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",               
            dirpath="Models/",
            filename=save_name,
        )
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("")  # logging results to a tensorboard
    
        trainer = pl.Trainer(
            # 모델이 학습을 수행할 최대 에포크 수
            max_epochs=Epoch,
            # 학습에 사용할 가속기 gpu 또는 cpu
            accelerator= 'gpu',
            # 학습에 사용할 GPU의 수
            devices= [0],
            # 모델 요약 정보를 출력할지 여부
            enable_model_summary=True,
            # 그래디언트 클리핑 값을 설정(그래디언트 폭주를 방지하기 위해 그래디언트의 최대 크기를 0.1로 제한)
            gradient_clip_val=0.1,
            # 한 에포크에서 사용할 최대 학습 배치의 수를 설정
            limit_train_batches=128,  
            # 학습 중 사용할 콜백 목록
            callbacks=[early_stop_callback, checkpoint_callback]
            #callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
            #logger=logger,
        )
    
        tft = TemporalFusionTransformer.from_dataset(
            training,
            # 학습률 설정
            learning_rate=lr,
            # 은닉층 크기 설정
            hidden_size=h,
            # 어텐션 헤드 수 설정 (4 이하로 설정)
            attention_head_size=4,
            # 드롭아웃 비율 설정
            dropout=0.1,
            # 연속형 입력 변수에 대한 은닉층 크기 설정(hidden_size 이하로 설정)
            hidden_continuous_size=15,
            # 출력 크기 설정 (기본적으로 7개의 분위수 사용)
            output_size=7,
            # 손실 함수 설정
            loss=QuantileLoss(),
            # 로깅 간격 설정 (예: 10개 배치마다 로깅 수행)
            log_interval=10,
            # 검증 손실이 개선되지 않을 경우 학습률을 줄이기 위한 기다림 횟수 설정
            reduce_on_plateau_patience= 20,
        )
    
        ################################# Train #################################
    
        # fit network
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
#%%
