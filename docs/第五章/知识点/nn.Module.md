```torch.nn.Module```是PyTorch中构建神经网络的核心基类，几乎所有```模型```、```层```、```损失函数```都继承自它。理解它是掌握PyTorch的关键
---
## Module是什么
```nn.Module```是一个**容器**，它可以：
1. 存储**参数**（```nn.Parameter```）和**子模块**（其他```nn.Module```）
2. 提供```forward()```定义前向计算
3. 自动管理参数的注册、设备前一、梯度、训练/评估模式等
一个最简单的Module
```
import torch
import torch.nn as nn

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear=nn.Linear(10,5)
    self.weight=nn.Parameter(torch.randn(5))

  def forwward(self,x):
    return self.linear(x)+self.weight
```
|赋值的类型|	行为|
|:---:|:---:|
|nn.Parameter|	注册到 self._parameters|
|nn.Module	|注册到 self._modules|
|其他（list/dict/tensor 等）	|存入普通 __dict__（不会被管理）|

```
model = MyModel()
model.parameters()      # 能拿到 linear.weight/bias 和自定义 weight
model.named_parameters() # 名字 + 张量
```

```
self.layers = [nn.Linear(10,10) for _ in range(3)]  # ❌ 不会注册
self.layers = nn.ModuleList([...])                   # ✅ 正确
self.layers = nn.ModuleDict({...})                   # ✅ 正确
self.layers = nn.Sequential(...)                     # ✅ 正确
```
### 关键方法
1. ```forward(*input)```
定义前向传播。**不要直接调用**```model.forward(x)```,**而要用**```model(x)```，因为```_call_```会触发hooks
```
def forward(self, x):
    return self.linear(x)
```
2. ```parameters()```/```named_parameters()```
返回所有可训练参数（递归包含子模块）
```
for name, p in model.named_parameters():
    print(name, p.shape)
```
3. ```children()```/```named_children()```
只返回**直接**子模块
```
for name, module in model.named_children():
    print(name, module)
```
4. ```modules()```/```named_modules()```
递归返回所有模块（包括自己）
5. ```buffers()```/```register_buffer()```
注册**不可训练**但需随模型保存/迁移的状态（如BatchNorm的running_mean）
```
self.register_buffer('running_mean', torch.zeros(features))
```
6. ```train()````/```eval()```
切换训练/评估模式，递归影响所有子模块（影响Dropout、BatchNorm等）
```
model.train()   # 训练模式
model.eval()    # 评估模式
```
7. ```to(device)```/```cuda()```/```cpu()```/```float()```
递归迁移所有参数和buffer
```
model.to('cuda')
model.half()   # fp16
```
8. ```state_dict()```/```load_state_dict()```
序列化/加载参数和buffer（不含结构）
```
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))
```
9. ```zero_grad()```
清空所有参数梯度
```optimizer.zero_grad()```# 推荐用优化器的
10. ```apply(fn)```
递归对所有子模块应用函数
```
model.apply(lambda m: nn.init.xavier_uniform_(m.weight) 
            if isinstance(m, nn.Linear) else None)
```
### 常用的内置Module子类
|类别|示例|
|:---:|:---:|
|线性层|```nn.Linear``` ```nn.Bilinear```|
|卷积层|```nn.Conv2d``` ```nn.ConvTranspose2d```|
|循环层|```nn.LSTM``` ```nn.GRU``` ```nn.RNN```|
|归一化|```nn.BatchNorm2d``` ```nn.LayerNorm``` ```nn.GroupNorm```|
|激活|```nn.ReLU``` ```nn.GELU``` ```nn.Sigmoid```|
|Dropout|```nn.Dropout``` ```nn.Dropout2d```|
|容器|```nn.Sequential``` ```nn.ModuleList``` ```nn.ModuleDict```|
|损失|```nn.CrossEntropyLoss``` ```nn.MSELoss```|
### 容器类Module对比
|容器|	是否注册子模块|	是否有 forward	适用场景|
|:---:|:---:|:---:|
|nn.Sequential|	✅|	✅顺序执行|	简单顺序网络|
|nn.ModuleList|	✅|	❌	|循环中存放多个层|
|nn.ModuleDict|	✅|	❌	|按名字索引的层|
|普通 list     |❌|	❌	|仅存 Python 对象|
```
# Sequential：自动 forward
net = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# ModuleList：需要自己写 forward
self.layers = nn.ModuleList([nn.Linear(10,10) for _ in range(3)])
def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return x
```
### Hook机制
Module 支持在 forward 前后插入钩子，用于调试、可视化、梯度裁剪等。
```
# 前向 hook
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__} output shape: {output.shape}")

model.linear.register_forward_hook(hook_fn)

# 反向 hook
model.linear.register_full_backward_hook(lambda m, gi, go: print(gi))
```
常见 hook：
+ register_forward_hook
+ register_forward_pre_hook
+ register_full_backward_hook
+ register_full_backward_pre_hook
### 常见易错点
1. 忘记 super().__init__() → 参数无法注册，报错。
2. 用普通 list 存层 → 参数不在 state_dict() 中，不会迁移到 GPU。
3. 直接调用 forward() → 绕过 hooks；应使用 model(x)。
4. eval() 后忘记 train() → Dropout/BatchNorm 行为错误。
5. load_state_dict 参数名不匹配 → 用 strict=False 或检查 key。
6. 共享参数：同一 nn.Linear 实例赋给两个属性会共享参数，这是特性但易误用。
7. nn.Parameter 与普通 tensor：只有前者会被 optimizer 更新。
## 核心机制：属性注册（```__setattr__```）
+ 这是Module最巧妙的设计，输入```self.xxx=...```时，Module 重写了```__setattr__```
+ 让PyTorch能够自动追踪和管理我定义的网络层和参数
### 核心目的：自动“收集”参数和子模块
当我定义神经网络时，会写很多```self.xxx=...```PyTorch需要知道：
+ 哪些变量是**需要训练的参数**（比如权重```weight```、偏置```bias```）
+ 哪些变量是**子网络层**（比如```nn.Linear``` ```nn.Conv2d```）
+ 哪些变量只是**普通的临时变量**（比如一个记录准确率的列表，说着一个普通的Tensor）

```__setattr__```就是那个分拣员，执行赋值时，他会**自动**根据**类型**把东西放到不同的“篮子”里
+ ```nn.Parameter```->放进```self._parameters```篮子（**可训练数据**）
+ ```nn.Module```->放进```self._modules```篮子（**子模块**，比如层）
+ 其他类型->放进普通的```_dict_```篮子(**不会被当做网络的一部分，不参与训练**)
### 好处（为什么这么设计）
+ 没有这个机制->手动维护一个列表，记录所有的权重和层，非常容易出错
+ 有了这个机制->像写Python类一样写代码，剩下的交给PyTorch：
  + ```model.parameters()```**能自动找到所有参数**，PyTorch会遍历字典（用于传给优化器```optimizer```）
  + ```model.named_parameters()```**能拿到名字和张量**，比如定义```self.linear=nn.Linear(...)```,方便打印、调试或保存模型
  + ```model.cuda()```或```model.to(device)```**能自动搬家**，把模型移动到GPU时，PyTorch只需遍历```_parameters```和```_modules```就能把里面所有东西都搬到GPU上
  + ```model.state_dict()```**能自动保存权重**，保存模型时，也是通过这个注册机制，自动把所有参数提取出来存成字典
### 总结
这个“注册”机制是用来**建立层级关系**的，让PyTorch知道：
+ **谁是网络的一部分**（需要训练、需要保存、需要搬到GPU）
+ **谁只是普通的Python变量**（不需要被管理）
它是PyTorch实现“自动微分”、“自动保存加载”、“自动设备迁移”的基础设施







