from  datetime import datetime
import pandas as pd 
import torch 
from torch.utils.data import Dataset,DataLoader
from alive_progress import alive_bar
import numpy as np
from torch.utils.data import random_split
import pickle as pkl

# here we do

#     1) save the data in different time slots 
#     2) prepare series data out of this sloted data    "in one day interval only"     and try to save if possible


def sloted_data(raw_data,Time_slot):

    # here we collect the comapany list and then devide the day data as sloted data
    if Time_slot==0:

        # pass entire row for data
        print(" time slot 0 need to be performed")
        
    elif Time_slot>0:
        
        data=[]

        start=0
        for i in range(0,int(len(raw_data)/Time_slot)):

            row=[]
            temp=raw_data[start:start+Time_slot]
            start+=Time_slot
            row.append(Time_slot)
            row.append(temp['open'].values[0])
            row.append(temp['close'].values[-1])
            row.append(max(temp['high']))
            row.append(min(temp['low']))
            row.append(temp["volume"].sum())
            data.append(row)
        processed_data=pd.DataFrame(data,columns=['time_slot',"open",'high',"low","close","volume"])

    return processed_data

def prepare_series(data,window):
    # here we calculate the closeing price 
    tempx=[]
    tempy=[]
    for i in range(int(len(data)/window)*window-window):
        tempx.append(np.array(data[i:i+window].to_numpy()).reshape((1,-1)))
        tempy.append(np.array(data['close'][i+window]))

    data_x=np.array(tempx).squeeze(1)
    data_y=np.array(tempy).reshape(-1,1)
    return data_x,data_y

def get_data(path,Time_slot,window,batch_size):


    class my_dataset(Dataset):
        def __init__(self,path,Time_slot,window):
            if path!=None:
                data=pd.read_csv(path)

                data[['date','time']]=data['date'].str.split(expand=True)
                dates=[]
                Time_slot=3
                window=10
                for i in data['date']:
                    if i.split()[0]  not in dates:  
                        dates.append(i.split()[0])
                data_x=[]
                data_y=[]

                with alive_bar(len(dates)) as bar:
                    for i in dates:

                        temp=sloted_data(data[data['date']==i],Time_slot)
                        o=prepare_series(temp,window)
                        data_x.append(o[0])
                        data_y.append(o[1])

                        bar()

                self.data_x=torch.tensor(np.vstack(data_x),requires_grad=True,dtype=torch.float32)
                self.data_y=torch.tensor(np.vstack(data_y),requires_grad=True,dtype=torch.float32)

                pkl.dump(self.data_x,open('saved_data/train_data_x.pkl',"wb"))
                pkl.dump(self.data_y,open('saved_data/train_data_y.pkl',"wb"))
            
            else:
                self.data_x=pkl.load(open('saved_data/train_data_x.pkl','rb'))
                self.data_y=pkl.load(open('saved_data/train_data_y.pkl','rb'))

        
        def __len__(self):
            return self.data_x.shape[0]

        def __getitem__(self, index):
            return self.data_x[index],self.data_y[index]
    
    dataset=my_dataset(path,Time_slot,window)

    train_data,test_data=random_split(dataset,[0.5,0.5])
    train_DataLoader=DataLoader(train_data,batch_size)
    test_DataLoader=DataLoader(test_data,batch_size)

    return  train_DataLoader,test_DataLoader

