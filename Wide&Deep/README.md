Wide & Deep是推荐系统的经典模型

原始论文中已经说得很清楚了，笔者在这里说一说自己对Wide & Deep的理解：

从模型内容Pooling的思路出发，如果Wide和Deep部分同时喂相同的特征，Wide & Deep是可以用残差网络解释的，并且从一定程度上说，Wide & Deep 自带残差网络。为什么？

F(x) = f(x) + x， f(x)可视为DNN（非线性），x可视为LR（线性）

研究者发现：深层神经网络会让模型退化的主要原因是激活函数ReLU的存在。数据经过激活函数之后是无法再反推回到原始状态的，整个过程是不可逆的。即当使用ReLU等激活函数时，会导致信息丢失：

低维（2维）的信息嵌入到n维的空间中，并通过随机矩阵T对特征进行变换，再加上ReLU激活函数，之后在通过 T^(-1)（反变换）进行反变换。当n=2，3时，会导致比较严重的信息丢失，部分特征重叠到一起了；当n=15到30时，信息丢失程度降低。这是因为非线性激活函数（Relu）的存在，每次输入到输出的过程都几乎是不可逆的（信息损失），所以很难从输出反推回完整的输入。

![image](https://user-images.githubusercontent.com/68730894/115326678-3b8fba80-a1c0-11eb-8d74-f4c277a113d8.png)
![image](https://user-images.githubusercontent.com/68730894/115326683-3fbbd800-a1c0-11eb-945c-e6837d54005d.png)

从数学公式看，如果没有非线性激活函数，残差网络存在与否意义不大。如果残差网络存在，则只是做了简单的平移：
![image](https://user-images.githubusercontent.com/68730894/115326716-5104e480-a1c0-11eb-86a6-9783845b408c.png)

增加非线性激活函数之后，上述式子发生改变，模型的特征表达能力大幅提升。这也是为什么Residual Block有2个权重（W_1,W_2）的原因。

![image](https://user-images.githubusercontent.com/68730894/115326744-5eba6a00-a1c0-11eb-8d64-7a0cd29f1c28.png)

残差网络升级扩展，数学上证明：
原始残差网络

![image](https://user-images.githubusercontent.com/68730894/115329746-557fcc00-a1c5-11eb-859e-a0871c94a291.png)

增加非线性激活

![image](https://user-images.githubusercontent.com/68730894/115329737-50228180-a1c5-11eb-9b83-280ac2fbe376.png)


为了实现一直堆叠网络而不发生网络退化的需要，何凯明让模型内部结构具备恒等映射能力：将上一层（或几层）之前的输出与本层计算的输出相加，可以将求和的结果输入到激活函数中做为本层的输出。

![image](https://user-images.githubusercontent.com/68730894/115329308-b6f36b00-a1c4-11eb-8663-ea1adf866b6e.png)

此外，FM也自带残差网络效果
![image](https://user-images.githubusercontent.com/68730894/115329578-13568a80-a1c5-11eb-9b34-871b845fdfd9.png)

工程上来说，该模块在原始论文中提到由特别强的记忆能力，笔者认为，这应该是是依靠LR强大的“评分卡”实现的，此外，该模块也赋予模型强大的可解释性能力

在不追求复杂模型那么一点点准确率的前提下，DNN部分作为LR的补充，兼具可解释性和高阶特征提取能力。
