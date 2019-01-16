# @TIME : 2019/1/15 上午12:31
# @File : pytorch_classification.py

import torch
import matplotlib.pyplot as plt

# 获取数据, 2个点簇, 注意x, y 的数据格式
n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data, 1)
x1 = torch.normal(-2*n_data, 1)
y0 = torch.zeros(100)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

print(x)
print('---')
print(y)

# 建立模型
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2) #几个类别就几个output
print(net)

# 训练模型
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()   # 画图
plt.show()
for t in range(100):
    out = net(x)
    loss = loss_func(out, y)
    print('loss:', loss)
    print('opeimizer:', optimizer)
    optimizer.zero_grad()
    loss.backward()
    print('loss_2', loss)
    optimizer.step()
    print('optimizer_2',optimizer)
    print('----------------------')

    # 接着上面来
    if t % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()

# 画图
