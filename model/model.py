from torch import nn

import torch


class Live_trading(nn.Module):

    def __init__(self,lstm_input_size,lstm_hidden_size,num_layers):

        super().__init__()

        self.lstm=nn.LSTM(lstm_input_size,lstm_hidden_size,num_layers,batch_first=True)

        self.seq=nn.Sequential(
            nn.Linear(lstm_hidden_size,lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size,1),
            nn.ReLU()
        )

    def forward(self,x):

        op,(o1,o2)=self.lstm(x)
        temp=self.seq(op)
        return temp
    
    