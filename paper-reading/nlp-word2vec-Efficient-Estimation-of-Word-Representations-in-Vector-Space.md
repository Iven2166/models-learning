Efficient-Estimation-of-Word-Representations-in-Vector-Space
---

paper: https://arxiv.org/pdf/1301.3781.pdf

---
# Content 目录
1. Introduction
- 1.1 Goals of the paper
- 1.2 Previous work
2. Model architectures
- 2.1 Feedforward neural network net language model (NNLM)
- 2.2 Recurrent neural net language model (RNNLM)
- 2.3 Parallel training of neural network
3. New log-linear model 
- 3.1 Continuous bag-of-words model
- 3.2 Continuous skip-gram model
4. Results
- 4.1 Task description
- 4.2 Maximization of accuracy
- 4.3 Comparison of model architectures
- 4.4 Large scale parallel training of models
- 4.5 Microsoft research sentence completion challenge
5. Examples of the learned relationships
6. Conclusion
7. Follow-up work

---
摘要
- 提出了两个模型，用于从非常大的数据集中计算单词的连续向量表示
- 效果检测：（1）词汇相似性 （2）与过往的神经网络模型相比
- 高准确、更低成本（用不到一天的时间从 16 亿个单词中学习高质量的词向量）
- 我们表明这些向量在我们的测试集上提供了最先进的性能，用于测量句法(syntactic)和语义词(semantic)的相似性

# 1. Introduction

介绍了现状
- 当下（2013年）很多模型将单个词汇视为最小单元，没有体现单词之间的相关性(因为在词汇表里呈现为index)
    - 具备一定合理性：模型简洁、稳定性、能够在很大的数据集
    - N-gram模型是一个很好的例子，理论上可以训练数据全集（trillions of words）
    
- 这样简洁的模型依旧有局限
    - 由于需要大量数据，比如语音识别这种数据集较小的领域应用不佳
    - 因此，用简易的技术堆积并非能够带来巨大改进
    
- 最佳构想是对词的表示，比如神经网络模型优于N-gram模型

## 1.1 Goals of the paper
- 本文主要目的：
    - 从巨大数据集（数十亿的词、数百万的词汇量）之中构造高质量的词汇向量表示
    - 对比现状：大部分现存的技术，只能够处理数千万的词、单词向量的modest维度只有50-100

- 词汇向量表示的质量 - 近期技术 [[20]](#20):
    - 相似词汇更加相近
    - 词汇有不同的相似程度（可区分、衡量）
    - 在 context of inflectional languages 已有体现——一个名词会有多个结尾词，一般从原有的向量空间里找类似词汇，也可以找到类似的结尾词
    
- 词汇的表示的相似度并不止于（1）简单的词汇含义/语法相似，还有（2）词向量的计算：$vector("king") - vector("man") + vector("woman)$ 约等于 $vector("queen")$ —— 相当于空间上向量之间的关系


```In this paper, we try to maximize accuracy of these vector operations by developing new model architectures that preserve the linear regularities among words. We design a new comprehensive test set for measuring both syntactic and semantic regularities1, and show that many such regularities can be learned with high accuracy. Moreover, we discuss how training time and accuracy depends on the dimensionality of the word vectors and on the amount of the training data.```

## 1.2 Previous work
- NNLM：线性映射层 以及 非线性隐藏层 被用于学习 词向量表示以及统计性语言模型
- 词向量能够显著提升并且简化 NLP 应用。
- 但现有的技术，计算量太复杂

# 2. Model architectures

有很多模型应用于词表示：包括了 Latent Semantic Analysis (LSA) 和 Latent Dirichlet Allocation (LDA)

本文主要做 词的分布式表示 (distributed representations of words) 
- 应用神经网络优于LSA：在保持词汇之间的线性规则 (linear regularities)
- LDA在大数据集上计算复杂度过高

为了对比模型: 1. 模型的参数个数 2.提升准确率，同时保持较低的计算复杂度

因此，模型的训练复杂度以该比例增长： O = E × T × Q，所有模型应用梯度随机下降以及反向传播算法。
- E: 训练epoch数，一般是 3～50
- T: 训练集的词汇数，可达到十亿级别(billions)
- Q: 模型的自定义参数


## 2.1 Feedforward neural network net language model (NNLM)

NNLM 应用于 [[1]](#1)

$y = b + WX + U tanh(d + HX)$

- y: V * 1
- b: V * 1
- W: V * (n*m)
- X: (n*m) * 1
- U: V * h
- H: h * (n*m)


$WX$ 为可选择训练的部分；$d + HX$ 为隐藏层的计算，再通过U计算输出

---
$$Q = N × D + N × D × H + H × V$$

- 复杂度的计算
  - 映射部分:$N × D$ 应该是从全部词汇的词向量表里寻找出这N个词的映射，共用统一的词向量表示;即N个词，找到D维度的词向量
  - 到隐藏层的连接：有$N × D$个维度的词向量拼接，需要连接$H$个隐藏层因子，复杂度为$N × D × H$
  - 隐藏层到最终该单词的预测，需要产出全部词汇的概率推测(V个)：$H × V$
  
---
模型的应用——降低复杂度
- hierarchical softmax
- 词汇表被表示为Huffman二叉树：vocabulary is represented as a Huffman binary tree
- 计算复杂度被降低为 $log2(V)$


## 2.2 Recurrent neural net language model (RNNLM)

## 2.3 Parallel training of neural network

- DistBelief [6] 的大规模分布式框架
- 小批量异步梯度下降和称为 Adagrad [7] 的自适应学习率

# 3 New Log-linear Models
- 上一节的主要发现是复杂度较多地来源于模型中的非线性隐藏层，而这恰好是神经网络的核心，不宜改变
- 探索更简单的模型，可能无法像神经网络那么准确地进行表示，但降低计算复杂度并且能够在大数据集上进行训练

## 3.1 Continuous Bag-of-Words Model (CBOW)
- 架构类似于前馈 NNLM
- 去除了非线性隐藏层
- 所有单词共享投影层（不仅仅是投影矩阵）； 因此，所有单词都被投影到相同的位置（它们的向量被平均）
- input是四个历史词、四个未来词，用于预测中间词
- 计算复杂度： $Q = N × D + D × log2(V)$
  - N个输入单词、D个向量维度、huffman二叉树的log2(V)
  - 注意：这里没有隐藏层，直接就是通过一个softmax()函数来归一化概率

## 3.2 Continuous Skip-gram Model
- 使用每个当前单词作为具有连续投影层的对数线性分类器的输入，并预测当前单词前后一定范围内的单词
- 我们发现增加范围可以提高结果词向量的质量，但也会增加计算复杂度。
- 由于距离较远的词通常与当前词的相关性低于与当前词的相关性，因此我们通过在训练示例中从这些词中抽取较少的样本来给予较远的词更少的权重。
- 计算复杂度： $Q = C × (D + D × log2(V))$

# 4 Results
- 词汇之间的关系: big - biggest 
- 词汇组的关系: word pairs: big - biggest and small - smallest
- 可以用简单的代数计算寻找: $vector("smallest")= vector(“biggest”)-vector(“big”) + vector(“small”)$

## 4.1 Task Description
如何衡量任务准确率？
- Overall, there are 8869 semantic and 10675 syntactic questions.
  
- 每个类别分两步创建
  - (1) 手工构建相似词对列表(a-b, c-d, ...)
  - (2) 通过两个单词对构成大问题列表(a-b _ c-d, ...)

比如，列举了美国68个大城市以及其州（单词对），再随机地抽取两个单词对构成 2.5k (68*67/2=2278) 个问题

- 问题评估
  - 该词的最近距离的单词为问题的答案，才算对（同义词不算对）
  - 评估 semantic, syntactic 两个方面
  
---
semantic理解为单词对的对应关系，syntactic对应的是时态变化等语法类相似度 [参考](https://davideliu.com/2020/03/16/word-similarity-and-analogy-with-skip-gram/)

`Word analogy evaluation has been performed on the Google Analogy dataset which contains 19544 question pairs, (8,869 semantic and 10,675 syntactic questions)and 14 types of relations (9 morphological and 5 semantic). A typical semantic question can have the following form: rome is to italy as athens is to where the correct answer is greece. Similarly, a syntactic question can be for example: slow is to slowing as run is to where the correct answer is clearly running.`

## 4.2 Maximization of Accuracy

- 我们使用谷歌新闻语料库来训练词向量。 该语料库包含大约 6B 个标记。 
- 我们将词汇量限制为 100 万个最常用的单词。 
- 显然，我们正面临时间受限的优化问题，因为可以预期使用更多数据和更高维的词向量都会提高准确性。 
- 为了估计模型架构的最佳选择以快速获得尽可能好的结果，我们首先评估了在训练数据子集上训练的模型，词汇限制在最频繁的 30k 词。

![img_1.png](img_1.png)

## 4.3 Comparison of Model Architectures


# 6. Conclusion

![img.png](img.png)
- 本文中的主要工作：在一系列的质量句法和语义语言任务(syntactic and semantic language tasks)上，研究了不同模型下的词向量表示的质量。

- 模型架构相对简单，提升计算速度，从而纳入了更多的训练样本。
    - 由于计算复杂度低得多，可以从更大的数据集中计算出非常准确的高维词向量
    - 与NNLM、RNNLM对比（文中的第2章）
    
- 应用DistBelief 分布式框架：训练 CBOW 和 Skip-gram，即使在具有一万亿个单词的语料库上也可以建模，基本上是无限大小的词汇表



<a id="1">[1]</a>
Y. Bengio, R. Ducharme, P. Vincent. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137-1155, 2003.
-- https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

<a id="20">[20]</a> 
T. Mikolov, W.T. Yih, G. Zweig. Linguistic Regularities in Continuous Space Word Representations. NAACL HLT 2013.
