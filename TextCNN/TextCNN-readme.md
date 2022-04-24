# 原作 + 原理推导

模型原理 (原作 `2.Model`)：

- 构造词（句）的向量表示：文章中使用到了 word2vec 训练的词向量，

$x_{i} \in \mathbb{R}^{k}$: 句子中第 $i$ 个词，向量具备 $k$ 个维度

- 在一定窗口长度下构造句子的部分向量，横向拼接，xi:i+j代表concat了xi到xi+j的单词向量

$x_{1:n}= x_{1} \oplus x_{2} \oplus . . . \oplus x_{n}$

- 构造卷积filter，w∈ R hk，用于前一步的h个单词向量，产生新特征ci；函数f 是非线性函数 hyperbolic tangent

ci = f(w · xi:i+h−1 + b)

- 经过全部窗口，产生n-h+1个ci，再对ci进行拼接，获取c

c = [c1, c2, . . . , cn−h+1], c ∈ R n−h+1

- 采用max-pooling处理，c帽提取最显著的特征（由于在句子分类任务中，提取最显著的特征进行分类即可，所以max比较有效）
`The idea is to capture the most important feature—one with the highest value—for
each feature map. This pooling scheme naturally
deals with variable sentence lengths.`

- 其它因素：单词向量的"渠道(channels)"： （1）单词向量保持不变 （2）单词向量依据反向传播进行迭代更新

正则化 (原作 `2.1 Regularization`)

- dropout: 对倒数第二层(`penultimate layer`, z = [ˆc1, . . . , cˆm] )进行遮盖处理，在前向传播中(forward propagation)不使用 y = w · z + b， 而是 y = w · (z ◦ r) + b 梯度仅在不遮盖的单元里进行反向传播更新参数。
r ∈ Rm is a ‘masking’ vector of Bernoulli random variables with probability p of being 1.

- `At test time, the learned weight vectors are scaled by p such that wˆ = pw, and wˆ is used (without dropout) to score unseen sentences. We additionally constrain l2-norms of the weight vectors by rescaling w to have ||w||2 = s whenever ||w||2 > s after a gradient descent step.`

# 其他参考

- TextCNN是很适合中短文本场景的强baseline，但不太适合长文本
  - 因为卷积核尺寸通常不会设很大，无法捕获长距离特征
  - 同时max-pooling也存在局限，会丢掉一些有用特征
- TextCNN和传统的n-gram词袋模型本质是一样的，它的好效果很大部分来自于词向量的引入[3]，因为解决了词袋模型的稀疏性问题。

在TextCNN的实践中，有很多地方可以优化（参考这篇论文[1]）：

Filter尺寸：这个参数决定了抽取n-gram特征的长度，这个参数主要跟数据有关，平均长度在50以内的话，用10以下就可以了，否则可以长一些。在调参时可以先用一个尺寸grid search，找到一个最优尺寸，然后尝试最优尺寸和附近尺寸的组合

Filter个数：这个参数会影响最终特征的维度，维度太大的话训练速度就会变慢。这里在100-600之间调参即可

CNN的激活函数：可以尝试Identity、ReLU、tanh

正则化：指对CNN参数的正则化，可以使用dropout或L2，但能起的作用很小，可以试下小的dropout率(<0.5)，L2限制大一点

Pooling方法：根据情况选择mean、max、k-max pooling，大部分时候max表现就很好，因为分类任务对细粒度语义的要求不高，只抓住最大特征就好了

Embedding表：中文可以选择char或word级别的输入，也可以两种都用，会提升些效果。如果训练数据充足（10w+），也可以从头训练

蒸馏BERT的logits，利用领域内无监督数据

加深全连接：原论文只使用了一层全连接，而加到3、4层左右效果会更好[2]

[参考](https://mp.weixin.qq.com/s?__biz=MzAxMTk4NDkwNw==&mid=2247485854&idx=1&sn=040d51b0424bdee66f96d63d4ecfbe7e&chksm=9bb980faacce09ec069afa79c903b1e3a5c0d3679c41092e16b2fdd6949aa059883474d0c2af&token=793481651&lang=zh_CN&scene=21#wechat_redirect)

