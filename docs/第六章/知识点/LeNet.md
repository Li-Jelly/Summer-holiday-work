这一节介绍经典卷积神经网络LeNet
---
## 手写数字识别
+ 有一段时间神经网络不流行
### MNIST（当年的数据集）
+ LeNet网络提出时附带的数据集
  + 50000个训练数据（内存有几M）
  + 10000个测试数据
  + 图像大小28*28
  + 基本上就是黑白图
  + 已经标记好了
  + 10类
<img width="819" height="458" alt="图片" src="https://github.com/user-attachments/assets/7ad8d9f0-f2e4-4e81-ab87-11114e8c6163" />

<img width="814" height="444" alt="图片" src="https://github.com/user-attachments/assets/6395c18c-0ee3-4d81-9853-19828fde7b75" />

### LeNet是什么
<img width="840" height="473" alt="图片" src="https://github.com/user-attachments/assets/db87cf74-fd89-4215-9fe1-bc19b0bdb26e" />

+ 输入是32*32的image（加了填充）
<img width="152" height="188" alt="图片" src="https://github.com/user-attachments/assets/959cbd6b-bc56-4f81-9455-ff809acc3787" />

+ 第一层卷积层
  + 5*5的卷积层，输出通道是6（输出的高宽都是28）
  + 输出叫feature map
<img width="229" height="275" alt="图片" src="https://github.com/user-attachments/assets/82be8220-19f5-45f4-b1f5-371bc7c3bc17" />

+ 第一层池化层
  + 2*2的池化层
  + 把28*28变成了14*14
  + 通道数没有改变
<img width="177" height="188" alt="图片" src="https://github.com/user-attachments/assets/3804bfba-df00-44c7-bf9c-230a74e193a4" />

+ 第二层卷积层
  + 5*5的卷积层
  + 输入变成10*10
  + 通道数变成16
<img width="238" height="306" alt="图片" src="https://github.com/user-attachments/assets/ce98ad79-d3d7-4f05-aeb6-de2807437b64" />

+ 第二层池化层
  + 同样的池化层
  + 高宽减半
  + 输入和通道数和输出不变
<img width="240" height="257" alt="图片" src="https://github.com/user-attachments/assets/c660c72e-c4a4-4179-bcb4-584c1ebc49a0" />

+ 第一个全连接层
  + 把前面的池化层拉成一个向量
  + 输出是120
<img width="51" height="220" alt="图片" src="https://github.com/user-attachments/assets/7b5988bb-9c6c-49f4-af7b-e36416aeb447" />

+ 第二个全连接层
  + 输出是64
<img width="36" height="132" alt="图片" src="https://github.com/user-attachments/assets/66ae3f16-cf14-4177-91dc-46c2898d66ce" />

+ 高斯层（也是一个全连接层）
  + 输出是10
  + 再做softmax得到一个概率
<img width="49" height="88" alt="图片" src="https://github.com/user-attachments/assets/ab378747-23cd-42d7-bd61-b0d5194db5c8" />

## 总结
+ LeNet是早期成功的神经网络
+ 先使用卷积层来学习图片空间信息
+ 池化层降低空间敏感度
+ 然后使用全连接层来转换到类别空间，得到实类
+ 从图片到类别的映射

## 代码部分
### LeNet(LeNet-5)由两个部分组成：卷积编码器和全连接层密集块
```
import torch
from torch import nn
from d2l import torch as d2l

class Reshape(torch.nn.Module):
  def forward(self,x):
    return x.view(-1,1,28,28)

net=torch.nn.Sequential(
  Reshape(),nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
  nn.AvgPool2d(kernel_size=2,stride=2),
  nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
  nn.AvgPool2d(kernel_size=2,stride=2),nn.Flatten(),
  nn.Linear(16*5*5,120),nn.Sigmoid(),
  nn.Linear(120,84),nn.Sigmoid(),
  nn.Linear(84,10))
```
+ 想用nn.Sequential
  + 自定义一个类，Reshape
    + 把x放成一个批量数不变，通道数变成1，高宽28（批量数，通道数，高，宽）
+ 把1*28*28的图片放进卷积层里面
  + 输入通道是1
  + 输出通道是6
  + 核是5*5
  + 填充是2
    + 最早LeNet是32*32（也是在两侧填充2）
    + 这里输入的是28*28（把边缘删掉了），所以手动补上2
  + 要得到非线性性，在卷积后面加上Sigmoid的激活函数
+ 用均值池化层
  + 核是2（可以不用写kernel=2，因为是第一个参数，直接写2，会自动识别kernel）
  + 步幅是2，不进行重叠
+ 卷积层
  + 输入是6
  + 输出从6变到16（变更多了）
  + kernel——size是5
  + 不做填充
  + 后面跟一个Sigmoid的激活函数
+ 第二个均值池化层
  + 超参数和前面相同（核2，步幅2）
  + 最后卷积层出来是4-D，要把最后的通道数、高、宽变成一维的向量，输入到多层感知机中
    + nn.Flatten()是第一维（批量保持住，后面全拉成向量）
+ 第一层输出是120
  + 激活一下
+ 120是输入（16*5*5）
  + 120降到84
+ 84是输入
  + 降到10
+ 10是类别
+ 后面是有两个隐藏层的多层感知机
+ 前面是两个卷积层
  + 每个卷积层后面有一个激活层
  + 每个卷积层后面有一个池化层
```
X=torch.rand(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
  X=layer(X)
  print(layer.__class__.__name__,'output shape: \t',X.shape)
```
+ 给一个随机的输入
+ nn.Sequential构造的，可以对里面的每一层做一次迭代
+ 输出每一层的尺寸
+ 自动的话，不太灵活，手动好一点
<img width="522" height="218" alt="图片" src="https://github.com/user-attachments/assets/971b0d1f-2628-44bc-9d22-968b21f13369" />

+ 第一个模块（卷积+激活+池化=通道数变为6，高宽减半）
  + 第一个卷积层把通道数变为6了，高和宽没有改变
    + 28*28，数字在图的最边缘，移一下就看不到了，所以要padding一下
    + 一般第一个卷积层的高和宽会减少，但这里加了padding，所以没有变
  + 激活曾不改变大小
  + 第一个池化层没有改变通道数
    + 高和宽改变了
+ 第二个模块（卷积+激活+池化）
  + 输入是上面的输出6*14*14
  + 输出是16*5*5
    + 高宽减了三分之二
  + 通道数从6变到16
+ 拉直以后就是MLP
  + MLP通过隐藏层的输出，把输出大小往下压
+ 卷积就是在把图变小变小变小，把通道变多变多变多
  + 每一个通道信息是一个空间的模式（边缘、锐化、模糊……）
  + 抽出被压缩的信息，放在不同的通道里面
+ MLP就是把这些不同通道的模式，通过一个多层感知机，训练到最后的输出
+ 通道数一直增加，高宽一直减小
  + 现在的神经网络可能会让通道数上千，让高宽变成1
+ 然后做全连接输出

### LeNet在Fashion-MNIST数据集上的表现
```
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)
```
### 对```evaluate_accuracy```函数进行轻微的修改
这里使用GPU（LeNet是课程中唯一一个能用CPU跑的网络）
```
def evaluate_accuracy_gpu(net,data_iter,device=None):
  """使用GPU计算模型在数据集上的精度。"""
  if isinstance(net,torch.nn.Module):
    net.eval()
    if not device:
      device=next(iter(net.parameters())).device
  metric=d2l.Accumulator(2)
  for X,y in data_iter:
    if isinstance(X,list):
      X=[x.to(device) for x in X]
    else:
      X=X.to(device)
    y=y.to(device)
    metric.add(d2l.accuracy(net(X),y),y.numel())
return metric[0]/metric[1]
```
+ 会实现一些手写的或者torch.nn的版本
  + 变成一个.eval()的Module
+ 如果device没有给定
  + 用net.parameters()
    + 把它的第一个参数拿出来，看它的device
+ Accumulator累加器
+ 对每个data_iter的X，y
  + 先移动到device上去
    + 如果是list就每个都挪一下
    + 如果是tersor就挪一次
+ 把X放在net中，得到输出，算一下accuracy（之前定义过了，存在d2l中）
+ 算一下y的元素个数
  + 所有分类正确的个数除以整个y的大小，得到accuracy
### train的函数在GPU上要做改动
```
def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
  """Train a model with a GPU(defined in Chapter 6)."""
  def init_weights(m):
    if typr(m)==nn.Linear or type(m)==nn.Conv2d:
      nn.init.xavier_uniform_(m.weight)
  net.apply(init_weights)
  print('training on',device)
  net.to(device)
  optimizer=torch.optim.SGD(net.parameters(),lr=lr)
  loss=nn.CrossEntropyLoss()
  animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],)
                         legend=['train loss', 'train acc', 'test acc'])
  timer, num_batches = d2l.Timer(), len(train_iter)
  for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```
+ 第六章的训练函数
  + 和第三章不一样的是多了device这个参数
+ 先初始化weight
  + 如果是全连接层/卷积层
    + 用xvaier_uniform这个定义好的操作来初始化
      + xvaier会根据输入输出的大小，在用随机初始化的时候让输入输出的方差差不多，不会在模型一开始的时候就炸掉或者变成0
+ apply：
  + 对每个parameter都run一下这个函数
+ 然后在device上训练（打印出来，防止没有跑在GPU上还没有发现）
+ 把net移动到device上
+ 这里直接用SGD
  + 给一个lr即可
+ loss(交叉熵损失函数)：
  + 一个多类分类问题（softmax没区别）
+ 动画效果
+ 对每次数据做迭代，每一次迭代拿一个batch出来
+ 梯度设零
+ 输入输出挪到GPU上
+ 做前向操作
+ 计算损失
+ 计算梯度
+ 再迭代
+ 最后再打印和动画一些东西
+ 核心和第三章基本没区别
  + 输入输出挪到GPU上（batch）
  + net work挪到GPU上
### 训练和评估LeNet-5模型
```
lr,num_epochs=0.9,10
train_ch6(net,train_iter,test_iteer,num_epochs,lr,d2l.try_gpu())
```
<img width="755" height="402" alt="图片" src="https://github.com/user-attachments/assets/ff7585d3-1b67-4374-8b5f-e193a9af1c69" />

+ 过拟合现象比MLP小
+ LeNet等价于一个受限的全连接层，模型少很多，意味着它的复杂度低，几乎没有过拟合现象（精度可以到0.83-0.84）
