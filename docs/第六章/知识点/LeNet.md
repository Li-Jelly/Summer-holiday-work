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
+ 然后使用全连接层来转换到类别空间













