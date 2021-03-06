
Attention Is All You Need

- [PDF](https://arxiv.org/pdf/1706.03762.pdf)
- [limu - Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE?spm_id_from=333.999.0.0)

- 其他参考：https://becominghuman.ai/multi-head-attention-f2cfb4060e9c
- https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/
- https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
- https://zhuanlan.zhihu.com/p/403433120


摘要

`基于注意力，摒弃递归和卷积`: 主要的`序列转录模型`基于复杂的循环或卷积神经网络，包括编码器和解码器。性能最好的模型还通过注意力机制连接编码器和解码器。我们提出了一种新的简单网络架构 Transformer，它完全基于注意力机制，完全摒弃了递归和卷积。

`任务上表现`: 对两个机器翻译任务的实验表明，这些模型在质量上更优越，同时更可并行化，并且需要的训练时间显着减少。我们的模型在 WMT 2014 英德翻译任务上达到了 28.4 BLEU，比现有的最佳结果（包括合奏）提高了 2 BLEU 以上。在 WMT 2014 英法翻译任务上，我们的模型在 8 个 GPU 上训练 3.5 天后，建立了一个新的单模型 state-of-the-art BLEU 得分 41.8，这只是最好的训练成本的一小部分。文献中的模型。我们表明，Transformer 通过成功地将其应用于具有大量和有限训练数据的英语选区解析，可以很好地推广到其他任务。

> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

1 Introduction

2 Background

3 Model Architecture

3.1 Encoder and Decoder Stacks

3.2 Attention

3.2.1 Scaled Dot-Product Attention

3.2.2 Multi-Head Attention

3.2.3 Applications of Attention in our Model

3.3 Position-wise Feed-Forward Networks

3.4 Embeddings and Softmax

3.5 Positional Encoding

4 Why Self-Attention

5 Training

6 Results

7 Conclusion


---

# 1 Introduction
要点
- 目前为止的研究：循环语言模型
- 算力约束：并行化能力差，sequential computation 的限制仍存在
- 注意力机制的应用：无需考虑在输入或输出序列中的距离；经常用于结合循环网络
- 论文的任务达成：提出Transformer模型，避免重复sequential架构，在翻译质量达到高水平


> 循环神经网络、长短期记忆 [13] 和门控循环 [7] 神经网络，尤其是已被牢固确立为序列建模和转导问题（如语言建模和机器翻译）的最先进方法 [35, 2 , 5]。 此后，许多努力继续推动循环语言模型和编码器-解码器架构的界限[38,24,15]。 
>
>循环模型通常沿输入和输出序列的符号位置考虑计算。 将位置与计算时间的步骤对齐，它们生成一系列隐藏状态 ht，作为先前隐藏状态 ht-1 和位置 t 的输入的函数。 `这种固有的顺序性质排除了训练示例中的并行化，这在更长的序列长度下变得至关重要，因为内存限制限制了示例之间的批处理`。 最近的工作通过分解技巧 [21] 和条件计算 [32] 显着提高了计算效率，同时在后者的情况下也提高了模型性能。 然而，顺序计算的基本约束仍然存在
> 
>注意机制已成为各种任务中引人注目的序列建模和转导模型的组成部分，允许对依赖项进行建模，而无需考虑它们在输入或输出序列中的距离 [2, 19]。 然而，除了少数情况[27]，这种注意力机制与循环网络结合使用。
> 
>在这项工作中，我们提出了 Transformer，这是一种避免重复的模型架构，而是完全依赖注意力机制来绘制输入和输出之间的全局依赖关系。 在八个 P100 GPU 上经过短短 12 小时的训练后，Transformer 可以实现更多的并行化，并且可以在翻译质量方面达到新的水平。

> Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].
> 
> Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of `sequential computation`, however, remains
> 
> Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.
> 
> In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

# 2 Background
整体讲清楚了该论文和其他论文（模型） 的关系，区别是什么，做了什么改进。

第一段：和卷积网络的关系
- 用卷积神经网络对长序列进行建模，每次都是看一个小窗口，而累计多层，才可以把两个比较远的位置联系起来。（比如每次看3*3的像素块，两个像素隔比较远，需要叠加多层卷积才可以看到联系）
但如果使用transformer里面的attention，看全部的序列。
- 而卷积模型比较好的一点是可以输出多个通道，那么transformer选择的是多头注意力机制。

第二段：自注意力机制
- 前人已经创造，在此并非创新

最后一段：
- transformer是第一个只依赖于注意力的、做encoder-decoder架构的模型


> 翻译要点：
> - 减少sequential computation 是各大模型的基础，以卷积神经网络作为基本构建块
> - 提取两个任意输入/输出位置的信号所需的操作量会随着距离增长而增长：ConvS2S-线性增长，ByteNet-对数增长。而Transformer做到常数
> - 由于平均注意力加权位置而降低了有效分辨率，我们使用多头注意力来抵消这种影响
> 
> 减少顺序计算的目标也构成了扩展神经 GPU [16]、ByteNet [18] 和 ConvS2S [9] 的基础，所有这些都使用卷积神经网络作为基本构建块，并行计算所有输入的隐藏表示和 输出位置。 在这些模型中，关联来自两个任意输入或输出位置的信号所需的操作数量随着位置之间的距离而增长，对于 ConvS2S 呈线性增长，而对于 ByteNet 则呈对数增长。 这使得学习远距离位置之间的依赖关系变得更加困难[12]。 在 Transformer 中，这被减少到`恒定数量`的操作，尽管由于平均注意力加权位置而降低了有效分辨率，我们使用多头注意力来抵消这种影响，如第 3.2 节所述。
> 
> 自注意，有时称为内部注意，是一种将单个序列的不同位置关联起来以计算序列表示的注意机制。 自注意力已成功用于各种任务，包括阅读理解、抽象摘要、文本蕴涵和学习任务无关的句子表示 [4, 27, 28, 22]。
> 
> 端到端记忆网络基于循环注意机制而不是序列对齐循环，并且已被证明在简单语言问答和语言建模任务中表现良好[34]。
> 
> 然而，据我们所知，Transformer 是第一个完全依赖自注意力来计算其输入和输出表示而不使用序列对齐 RNN 或卷积的转换模型。 在接下来的部分中，我们将描述 Transformer，激发自注意力并讨论其相对于 [17、18] 和 [9] 等模型的优势。

> The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a `constant number of operations`, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with `Multi-Head Attention` as described in section 3.2.
> 
> Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].
> 
> End-to-end memory networks are based on a recurrent attention mechanism instead of sequence aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].
> 
> To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

# 3 Model Architecture

> 大多数竞争性神经序列转导模型具有编码器-解码器结构 [5, 2, 35]。 这里，编码器将符号表示的输入序列 (x1, ..., xn) 映射到连续表示的序列 z = (z1, ..., zn)。 给定 z，解码器然后一次生成一个元素的符号输出序列 (y1, ..., ym)。 在每个步骤中，模型都是自回归的 [10]，在生成下一个时将先前生成的符号用作附加输入。
> 
> Transformer 遵循这种整体架构，对编码器和解码器使用堆叠的自注意力和逐点全连接层，分别如图 1 的左半部分和右半部分所示。

> Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.
> 
> The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

## 3.1 Encoder and Decoder Stacks

Encoder： 多头注意力机制，一个MLP，残差连接、layerNorm

展开讲了下LayerNorm
输入为2维时，[batch, n_feature]
- BatchNorm： 对每个小批量进行归一化（均值调为0，除以方差）
- LayerNorm： 对每个样本（行）做均值为0，方差为1

3D时，[batch, n_seq, n_feature] 列为 序列的长度， n_seq就是n，n_feature就是d（文中设定为512）
- BatchNorm：从特征维度切下去一个截面，那么是 [batch, n_seq]，但面临每个样本的时间步不同，而导致的均值和方差抖动较大；而如果输入的长度序列没在训练里出现过（比如很长，超过了常见的），那么不好预测
- LayerNorm：从batch维度切下去一个样本，那么是 [n_seq, n_feature]，均值和方差是在自己里面进行计算；不会出现上述问题。而后续也有文章提到，layerNorm有效在于梯度的更新。

Decoder：

- 在Encoder的输出之后，加入了多头注意力机制。 
  
- 跟Encoder类似，也是有残差连接、layerNorm；但还有 masked 掩码机制，来控制在预测t时刻的文本时（文本生成里面），不能看到t时刻及以后的输入，保证训练和预测时的行为是一致的。

接下来看每个子层是怎么定义的。


## 3.2 Attention

输出计算为值的加权和：输出的维度和value的维度是一样的。
权重是怎么来的：每一个value的权重，是由查询和value对应的key的相似度来决定的（不同注意力的模型有不同的算法）

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
> 
> 注意力函数可以描述为将查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。 输出计算为值的加权和，其中分配给每个值的权重由查询与相应键的兼容性函数计算。

### 3.2.1 Scaled Dot-Product Attention
维度query和key相同

$ query, key \in d_ k$

$ value \in d_ v $ 

计算query和全部key的点积，如果有n个query就有n个值
再除以维度长，再sortmax（就有n个非负的、和为1的权重），来得到value的权重，算到value上，就有输出了。
（点积计算相似度：[知乎](https://zhuanlan.zhihu.com/p/359975221#:~:text=%E5%90%91%E9%87%8F%E7%82%B9%E4%B9%98%EF%BC%9A%EF%BC%88%E5%86%85%E7%A7%AF,%E6%A0%87%E9%87%8F%E7%A7%AF%EF%BC%88Scalar%20Product%EF%BC%89%E3%80%82&text=%E4%BB%8E%E4%BB%A3%E6%95%B0%E8%A7%92%E5%BA%A6%E7%9C%8B%EF%BC%8C%E7%82%B9,%E5%A4%B9%E8%A7%92%E4%BD%99%E5%BC%A6%E7%9A%84%E7%A7%AF%E3%80%82) ）

> We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension $d_ k$, and values of dimension $d_ v$. We compute the dot products of the query with all keys, divide each by √ dk, and apply a softmax function to obtain the weights on the values.

实际上，query是一个矩阵 [n, d_k]，矩阵乘法能够非常好的并行。

$$ Attention(Q, K, V) = softmax( \frac {Q K^T} {\sqrt{d_k} }) V $$

> In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:
> 
> 在实践中，我们同时计算一组查询的注意力函数，并打包到矩阵 Q 中。键和值也打包到矩阵 K 和 V 中。 我们将输出矩阵计算为：

一般有两种注意力机制，一种是加性注意力机制，能够处理不等长的情况；另外一种叫做点积的注意力机制，和文章是一样的，除了文章多除了一个数之外（也就叫做scaled的原因）

为什么除以 根号$d_ k$ ，因为softmax让小的更小，大的更大。而 $d_ k$ 很大时，可能做出来的点积部分值会更大，让softmax把分布往两端拉得更开。而那些中间的值会很接近于0，导致梯度很小。

$\sqrt{d_k}$: 原因在于假定q和key都是均值为0方差为1的分布，所以点积是 $q_ i$ 与 $k_ i$ 的乘积之和是均值为0方差为 $d_ k$ 的数 

> The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
of $\frac {1}{d _k} $ . Additive attention computes the compatibility function using a feed-forward network with
a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
much faster and more space-efficient in practice, since it can be implemented using highly optimized
matrix multiplication code.

> While for small values of dk the two mechanisms perform similarly, additive attention outperforms
dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has
extremely small gradients . To counteract this effect, we scale the dot products by $\frac {1}{d _k} $. 


### 3.2.2 Multi-Head Attention

scaled Dot-Product Attention 并没有可学习的参数，仅是依靠点积来完成运算。而multi-head可以多学几个模式。

先让其投影到低维度，然后学习超参（通过h次）。

> Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2
> 
> 与使用 dmodel 维度的键、值和查询执行单个注意函数不同，我们发现将查询、键和值分别线性投影到 dk、dk 和 dv 维度上的不同学习线性投影是有益的。 然后，在每个查询、键和值的投影版本上，我们并行执行注意功能，产生 dv 维输出值。 这些被连接起来并再次投影，产生最终值，如图 2 所示

3.2.3 Applications of Attention in our Model

3.3 Position-wise Feed-Forward Networks

由于前面的attention已经汇聚(aggregate)了时序信息，所以后续的MLP只需要单个单个地对每个样本做线性变换即可。
而作为对比，RNN是在每个时间步，都要将前一个的信息合并目前的信息，一起进行计算。

3.4 Embeddings and Softmax

三个emb的权重实际上是一样，学习起来方便一些。

还在emb权重乘上 $\sqrt{d _{model}}$，因为emb越长由于norm的关系会导致里面的参数越小，乘后范围也在[-1,1]之间，和下方的sin、cos函数返回的[-1,1]范围一致。再和下面的position emb想加时会好点。

attention的权重不会带有时序信息。假设句子打乱，attention不变但是无意义。所以在输入里加入时序信息。

3.5 Positional Encoding


4 Why Self-Attention

解释Table1的复杂度比较，

sequential operation：在计算第几步之前需要计算多少步，越不用等越好。

maximum path length：两个位置的信息传播的步数（如何理解？下方原文）

>The third is the path length between long-range dependencies in the network. Learning long-range
dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the
ability to learn such dependencies is the length of the paths forward and backward signals have to
traverse in the network. The shorter these paths between any combination of positions in the input
and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare
the maximum path length between any two input and output positions in networks composed of the
different layer types
> 
> 第三个是网络中远程依赖关系之间的路径长度。 学习长程依赖是许多序列转导任务中的关键挑战。 影响学习这种依赖性的能力的一个关键因素是前向和后向信号必须在网络中遍历的路径长度。 输入和输出序列中任意位置组合之间的这些路径越短，就越容易学习远程依赖[12]。 因此，我们还比较了由不同层类型组成的网络中任意两个输入和输出位置之间的最大路径长度

各种模式的复杂度比较：
- attention 
    - 1. query和key的矩阵乘法，n^2*d
    - 2. 并行度 O(1)
    - 3. 
- recurrent
    - 1.
    
- convolutional：传递的距离是因为为卷积，一层层传上去，为取对数
- self-attention-restricted: 只对r范围内的进行建模，所以复杂度不用 n^2 而只需要 n*r，但信息传递就需要通过 n/r 步来调过来


5 Training

6 Results

# 7 Conclusion

在这项工作中，我们提出了 Transformer，这是第一个完全基于注意力的序列转导模型，用多头自注意力取代了编码器-解码器架构中最常用的循环层。

对于翻译任务，Transformer 的训练速度明显快于基于循环或卷积层的架构。 在 WMT 2014 英语到德语和 WMT 2014 英语到法语的翻译任务上，我们都达到了新的水平。 在前一项任务中，我们最好的模型甚至优于所有先前报道的集成。

我们对基于注意力的模型的未来感到兴奋，并计划将它们应用于其他任务。 我们计划将 Transformer 扩展到涉及文本以外的输入和输出模式的问题，并研究局部的受限注意力机制，以有效处理图像、音频和视频等大型输入和输出。 减少生成的顺序是我们的另一个研究目标。

# Appendix

多头注意力机制，`making` 单词会注意多个词之后的 `more difficult`

![img_2.png](img_2.png)