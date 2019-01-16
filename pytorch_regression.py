# @TIME : 2019/1/13 下午2:09
# @File : pytorch_regression.py


import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 数据
x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 构建神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
print('net', net)


# 训练网络
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

# for t in range(100):
#     prediction = net(x)
#     loss = loss_func(prediction, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# 可视化训练过程

plt.ion()  # 打开交互模式
plt.show()
for t in range(200):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.2)






