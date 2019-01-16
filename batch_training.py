# @TIME : 2019/1/16 上午8:19
# @File : batch_training.py


import torch
import torch.utils.data as Data
torch.manual_seed(1)    # reproducible

BATCH_SIZE = 8

# 1) 获得数据
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 2) 使用TensorDataset & DataLoader
dataset = Data.TensorDataset(x, y)
Loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=2
)

# 3) 打印一波
for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(Loader):
        print('Epoch:', epoch, '| Setp:', step, '| batch x:', batch_x.numpy(), '| batch y:', batch_y.numpy())
