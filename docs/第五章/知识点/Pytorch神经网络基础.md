## Pytorch 构建层和块
```
import torch
from torch import nn
from torch.nn import functional as F

net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

X=torch.rand(2，20)
net(X)
```
### 详解
```
import torch
from torch import nn
from torch.nn import functional as F
```
+ **第一行**：导入PyTorch主库，通常用别名```torch```
+ **第二行**：从```torch```中导入```nn```模块（神经网络模块），包含各种层、损失函数等
+ **第三行**：导入```torch.nn.functional```，通常用于激活函数、损失函数等（没有包括参数的函数）
```net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))```
+ 用```nn.Sequential```搭建一个**顺序容器**，数据会按顺序依次通过里面的层
  1. ```nn.Linear(20,256)```：全连接层（线性层），输入维度20，输出维度256，计算```y=XW+b```
  2. ```nn.ReLU()```：ReLU激活函数，```ReLU(x)=max(0,x)```，引入非线性
  3. ```nn.Linear(256,10)```：全连接层（线性层），输入256，输出10（比如对应10个类别）
```
X=torch.rand(2,20)
```
+ 生成一个形状为(2,20)的随机张量（2个样本，每个样本20个特征），值在[0,1)均匀分布
```
net(X)
```
+ 把输入```x```传入网络，前向传播：
  + ```x```：（2，20）->Linear(20->256)->(2,256)
  + ->ReLU->(2,256)
  + ->Linear(256->10)->(2,10)
+ 返回一个形状为（2，10）的张量（每个样本对应10个输出值）
+ 注意：这里没有赋值给变量，结果直接被丢弃，通常写成：
```
y=net(X)
```
总结：这是一个经典的**两层MLP**（多层感知机），常用于20维特征输入、10分类的任务
