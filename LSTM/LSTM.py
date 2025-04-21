#%%

import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# 모든 경고 메시지를 무시
warnings.filterwarnings("ignore")

#%%


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, sequence_length, num_layers, output_sequence, device):
        super(LSTM, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #self.fc = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size * sequence_length, output_sequence)
       

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)    # 초기 hidden state 설정하기.
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)    # 초기 hidden state 설정하기.
        
        out, _ = self.lstm(x, (h0, c0))  
        
        out = out.reshape(out.shape[0], -1)     # many to many 
        #out = out[:, -1, :]
        out = self.fc(out)

        return out

#%%

def seq_data(x, y, sequence_length, Term, device):
  
  x_seq = []
  y_seq = []

  for i in range((len(x) - sequence_length - Term) + 1):
    x_seq.append(x[i : i+sequence_length])
    y_seq.append(y[i+sequence_length : i+sequence_length + Term])

  return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device)


def inverse_standard(y, output, scaler, target_column):

    y = (y * scaler.scale_[target_column]) + scaler.mean_[target_column]
    output = (output * scaler.scale_[target_column]) + scaler.mean_[target_column]

    return y, output

#%%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(f'{device} is available')

#%%

###################################### Date Load ######################################

df = pd.read_csv('C:/Users/User/Desktop/Expfile/Exp_DLNM/Data/BDISet.csv', index_col=0)

df.index = pd.to_datetime(df.index)
target_column = df.shape[1]-1

print(df)
print(df.columns)

#%%

scaler = StandardScaler()
scaled = scaler.fit_transform(df)
df = pd.DataFrame(scaled, columns = df.columns, index = df.index)
print(df)

#%%

###################################### Data Preprocessing ######################################

Term = 12
sequence_length = 120

x = df.iloc[:, :target_column + 1].values
y = df.iloc[:, target_column].values
print(len(x))

x_seq, y_seq = seq_data(x, y, sequence_length, Term, device)

print(x_seq.shape)
print(y_seq.shape)

#%%

###################################### Data Split ######################################

x_train = x_seq[:812]
y_train = y_seq[:812]

x_valid = x_seq[812:]
y_valid = y_seq[812:]

print(x_train.size(), y_train.size())
print(x_valid.size(), y_valid.size())

plt.plot(y_valid[:, Term-1].cpu(), label='BDI')
plt.legend()

#%%

train = torch.utils.data.TensorDataset(x_train, y_train)

batch_size = 64

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)

#%%

###################################### Hyper Parameter ######################################

input_size = x_seq.size(2)
num_layers = 2
hidden_size = 32

#%%

model = LSTM(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   output_sequence = Term,
                   device=device).to(device)

#%%

###################################### Model Option ######################################

criterion = nn.MSELoss()

lr = 0.001
num_epochs = 1000
optimizer = optim.Adam(model.parameters(), lr=lr)


#%%

best = -1 * 10**9
early_stopping = 200
cnt = 0

n = len(train_loader)

for epoch in range(num_epochs):
    
    running_loss = 0.0
    prediction = []
    actual = []

    model.train()

    for data in train_loader:

        optimizer.zero_grad()  
        
        data_X, data_y = data                
        output = model(data_X) 
        loss = criterion(output, data_y)        # output 가지고 loss 구하고,
        loss.backward()                                 # loss가 최소가 되게하는 
        optimizer.step()                                # 가중치 업데이트 해주고,
        running_loss += loss.item()                     # 한 배치의 loss 더해주고,
    
    model.eval()

    with torch.no_grad():

        output = model(x_valid)

        # scale
        actual = y_valid[:, Term-1]
        prediction = output[:, Term-1]
        
        actual, prediction = inverse_standard(y=actual, output=prediction, scaler=scaler, target_column=target_column)
        actual = np.array(actual.cpu())
        prediction = np.array(prediction.cpu())
        
        rmse = np.sqrt(mean_squared_error(prediction, actual))
        mae = mean_absolute_error(prediction, actual)
        r2 = r2_score(prediction, actual)
        
    if best < r2:
        best = r2
        torch.save(model.state_dict(), f'LSTM{1}')
        cnt = 0
    else:
        cnt = cnt + 1
        if cnt == early_stopping:
            break

    if epoch%25 == 0:
        plt.plot(prediction, label='prediction', color='r')
        plt.plot(actual, label='actual', color='b')
        plt.legend()
        plt.show()
    
        print(f'epoch: {epoch} / Best R2: {best:.4f} / Test rmse: {rmse:.4f} / R2: {r2:.4f}')
  

#%%

# Test

model_name = f'LSTM1'

model = LSTM(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   output_sequence = Term,
                   device=device).to(device)
model.load_state_dict(torch.load(model_name))

model.eval()

actual = []
prediction = []

with torch.no_grad():

    output = model(x_valid)

    # scale
    actual = y_valid[:, Term-1]
    prediction = output[:, Term-1]
    
    actual, prediction = inverse_standard(y=actual, output=prediction, scaler=scaler, target_column=target_column)
    actual = np.array(actual.cpu())
    prediction = np.array(prediction.cpu())
    
    rmse = np.sqrt(mean_squared_error(prediction, actual))
    mae = mean_absolute_error(prediction, actual)
    r2 = r2_score(prediction, actual)

plt.plot(prediction, label='prediction', color='r')
plt.plot(actual, label='actual', color='b')
plt.legend()
plt.show()

print(f'epoch: {epoch} / Test rmse: {round(rmse, 4):.4f} / R2: {round(r2, 4):.4f}')
  



# %%
