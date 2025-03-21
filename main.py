from data_prep.prepare_data import get_data
from model.model import Live_trading
import torch
import pickle as pkl

comp_path="/mnt/6A8CB2D58CB29AD1/Live_trading_front_back/backend/data/ADANIGREEN_minute.csv"

comp_path=None
train_data,test_data=get_data(comp_path,3,10,128)




lstm_input_size=next(iter(train_data))[0].shape[1]
lstm_hidden_size=50
num_layers=1
epoch=5

model=Live_trading(lstm_input_size,lstm_hidden_size,num_layers).to("cuda")

loss=torch.nn.MSELoss().to("cuda")
opt=torch.optim.SGD(model.parameters())



for e in range(epoch):
    train_loss=0
    for j,i in enumerate(train_data):

        ypred=model(i[0].to("cuda"))
        diff=loss(ypred,i[1].to("cuda"))
        train_loss+=diff

        opt.zero_grad()
        diff.backward()
        opt.step()
        print("loss in the batch number ",diff)
    print(train_loss)
