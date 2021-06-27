import time
import torch
from torch import nn
import torchvision.models as models
import torch.utils.data as Data
def init_W(L):
    if type(L) == nn.Linear or type(L) == nn.Conv2d:
        nn.init.xavier_uniform_(L.weight)


def try_gpu(i=0):
    """
    -> 使用gpu, 找不到gpu 使用 cpu

    param:
        *``i :int`` gpu下标

    return:
       :torch.device

    """
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train_net(net, data, optimizer, criterion, init_func):
    """
    -> 训练网络

    param:
      *``net: nn.Model`` 要训练的网络
      *``data: torch.utils.data.TensorDataset`` 训练数据集
      *``optimizer: torch.optim`` 优化器
      *``criterion: torch.nn`` 损失函数
      *``init_func: funtion`` 用于初始化参数的方法

    return:
      train_losses: torch.device

    """
    net.to(try_gpu(0))

    train_losses = []
    epoch = 1
    net.train()
    
    loss_100 = 0
    for i, (x, y) in enumerate(data):
        t = time.time()
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        loss_100 += loss.item()
        if i % 100 == 0:
            print('time:{.4f}'.format(time.time()-t))
            print(f'{i}\n{loss_100 / 100}')
            loss_100 = 0
        
    return (train_losses)


# 组成dataset
def get_dataset(dataset, batch_size:int = 16, shuffle:bool = True, num_workers:int = 0):
    return Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        )

