**(1) 模型简述**

参考: [d2l-8.4. 循环神经网络](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html)

模型总体公式：

假设我们在时间步 $t$ 有小批量输入 

$\mathbf{X}_ {t} \in \mathbb{R}^{n \times d}$,
$\mathbf{H}_ {t} \in \mathbb{R}^{n \times h}$,
$\mathbf{W}_ {hh} \in \mathbb{R}^{h \times h}$,
$\mathbf{W}_ {xh} \in \mathbb{R}^{d \times h}$,
$\mathbf{b}_ {h} \in \mathbb{R}^{1 \times h}$,
$\mathbf{b}_ {q} \in \mathbb{R}^{1 \times q}$,

当前时间步隐藏变量由当前时间步的输入与前一个时间步的隐藏变量一起计算得出：

$$\mathbf{H}_ {t} = \phi ( \mathbf{X}_ {t} \mathbf{W}_ {xh} + \mathbf{H}_ {t-1} \mathbf{W}_ {hh}  + \mathbf{b}_ {h} )$$

对于时间步$t$，输出层的输出类似于多层感知机中的计算：

$$\mathbf{O}_ {t} = \mathbf{H}_ {t} \mathbf{W}_ {hq} + \mathbf{b}_ {q} $$


指标：困惑度(perplexity)：

我们可以通过一个序列中所有的 $n$ 个词元的交叉熵损失的平均值来衡量:

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_ {t-1}, \ldots, x_1)$$

其中$P$由语言模型给出，
$x_t$是在时间步$t$从该序列中观察到的实际词元。
这使得不同长度的文档的性能具有了可比性。

$$\exp\left(-\frac{1}{n} \sum_ {t=1}^n \log P(x_ t \mid x_ {t-1}, \ldots, x_ 1)\right).$$

* 在最好的情况下，模型总是完美地估计标签词元的概率为1。
  在这种情况下，模型的困惑度为1。
* 在最坏的情况下，模型总是预测标签词元的概率为0。
  在这种情况下，困惑度是正无穷大。
* 在基线上，该模型的预测是词表的所有可用词元上的均匀分布。
  在这种情况下，困惑度等于词表中唯一词元的数量。

小结

* 循环神经网络的隐状态可以捕获直到当前时间步序列的历史信息。
* 循环神经网络模型的参数数量不会随着时间步的增加而增加。
* 我们可以使用循环神经网络创建字符级语言模型。
* 我们可以使用困惑度来评价语言模型的质量。


**(2) 训练**

参考
- [d2l-8.7. 通过时间反向传播](https://zh.d2l.ai/chapter_recurrent-neural-networks/bptt.html)
- [d2l-8.5. 循环神经网络的从零开始实现](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html)


公式有空再整理

- 循环神经网络模型在训练以前需要初始化状态，不过随机抽样和顺序划分使用初始化方法不同。[见<随机采样和顺序分区>](https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html)
- 当使用顺序划分时，我们需要分离梯度以减少计算量。
  > 具体来说，当使用顺序分区时， 我们只在每个迭代周期的开始位置初始化隐状态。 由于下一个小批量数据中的第 𝑖 个子序列样本 与当前第 𝑖 个子序列样本相邻， 因此当前小批量数据最后一个样本的隐状态， 将用于初始化下一个小批量数据第一个样本的隐状态。 这样，存储在隐状态中的序列的历史信息 可以在一个迭代周期内流经相邻的子序列。 然而，在任何一点隐状态的计算， 都依赖于同一迭代周期中前面所有的小批量数据， 这使得梯度计算变得复杂。 为了降低计算量，在处理任何一个小批量数据之前， 我们先分离梯度，使得隐状态的梯度计算总是限制在一个小批量数据的时间步内。
- 在进行任何预测之前，模型通过预热期进行自我更新（例如，获得比初始值更好的隐状态）。
- 梯度裁剪可以防止梯度爆炸，但不能应对梯度消失
- 矩阵 $\mathbf{W}_ {hh} \in \mathbb{R}^{h \times h}$ 的高次幂可能导致神经网络特征值的发散或消失，将以梯度爆炸或梯度消失的形式表现。
- 截断是计算方便性和数值稳定性的需要。截断包括：规则截断和
  随机截断(实现方法是有一定概率在某个时间步上截断，因此不用全部计算；但总体的期望是1，相当于理论公式；某些时间步会有更大的梯度权重)。
- 为了计算的效率，“通过时间反向传播”在计算期间会缓存中间值。


> 梯度爆炸的公式推导: https://zhuanlan.zhihu.com/p/109519044
> ![img.png](./rnn-gradient-decrease1.png)



**(3)注意点**

QA：

- 为什么做字符而非预测单词？
  - 因为字符的vocab_size小，one-hot的向量也小。而单词则很大


---
- 参考
    - [深度学习之循环神经网络（RNN）](https://www.cnblogs.com/Luv-GEM/p/10703906.html)
    - [TensorFlow之RNN：堆叠RNN、LSTM、GRU及双向LSTM](https://www.cnblogs.com/Luv-GEM/p/10788849.html)
    - [循环神经网络之LSTM和GRU](https://www.cnblogs.com/Luv-GEM/p/10705967.html)
    - [rnn各类模型复现](https://github.com/spro/practical-pytorch)
    - [pytorch官方文章](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#creating-the-network)
    - [rnn-含有很多现实案例](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)


## 门控循环单元（gated recurrent units，GRU）

参考
- [9.1. 门控循环单元（GRU）](https://zh.d2l.ai/chapter_recurrent-modern/gru.html)

RNN需要解决的问题，可以由 GRU 和 LSTM来解决：
- 长期记忆 - 进行存储：早期观测值对预测所有未来观测值具有非常重要的意义。`我们希望有某些机制能够在一个记忆元里存储重要的早期信息`。 如果没有这样的机制，我们将不得不给这个观测值指定一个非常大的梯度， 因为它会影响所有后续的观测值。
- 选择遗忘 - 进行跳过：一些词元没有相关的观测值。 例如，在对网页内容进行情感分析时， 可能有一些辅助HTML代码与网页传达的情绪无关。 `我们希望有一些机制来跳过隐状态表示中的此类词元。`
- 逻辑中断 - 进行重置：序列的各个部分之间存在逻辑中断。 例如，书的章节之间可能会有过渡存在。我们希望重置我们的内部状态表示
- 其他需解决的：梯度消失和梯度爆炸


![img.png](./GRU.png)

- Reset Gate：重置门。 与 $H_ {t-1}$ 进行点积，由于是 $(0,1)$ 的范围，因此能够选择遗忘 $H_ {t-1}$，从而减少以往状态的影响
- Update Gate：更新门。 当 $Z_ {t}$ 偏向 1 时，则选择更多的 $H_ {t-1}$，即记住更长期的信息；如果偏向于0，则选择更多的候选隐状态，即含有 $X_t$的当前信息

小结：
- 门控循环神经网络可以更好地捕获时间步距离很长的序列上的依赖关系。
- 重置门有助于捕获序列中的短期依赖关系。
- 更新门有助于捕获序列中的长期依赖关系。
- 重置门打开时，门控循环单元包含基本循环神经网络；更新门打开时，门控循环单元可以跳过子序列。


## 长短期记忆网络（long short-term memory，LSTM） 
参考
- [9.2. 长短期记忆网络（LSTM）](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)


![img.png](./lstm-1.png)

小结
- 长短期记忆网络有三种类型的门：输入门、遗忘门和输出门。
- 长短期记忆网络的隐藏层输出包括“隐状态”和“记忆元”。只有隐状态会传递到输出层，而记忆元完全属于内部信息。
- 长短期记忆网络可以缓解梯度消失和梯度爆炸。

- 参考
  - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [为何能解决梯度消失问题？- Why LSTMs Stop Your Gradients From Vanishing](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html)
  - [为何能解决梯度消失问题？- 中文翻译](https://zhuanlan.zhihu.com/p/109519044)
  

## RNN簇的比较点

- RNN最长序列一般为几十，难以做到上百的长度（所以随机抽样即可，还可以避开overfit问题，不一定要因为序列的关联性而用顺序分区）
- GRU的更新门 $Z_ {t}$ 是加权求和（1-z与z），而LSTM的记忆单元 $C_ {t}$ 是加和得到的，而 $H_ {t}$ 是由输出门 $O_ {t}$决定是否重置
