*models-learning*

本文件夹用于积累对模型的学习及定期回顾。

- 整体的版图认知
- 各个模型的认知 & 实践 （一个模型一个文件夹，需要包括md 及 ipynb模型训练文件）

# 重构V1

# 统计学习方法

框架基于《统计学习方法-李航》，对经典的机器学习方法进行原理学习、复现。




# 深度学习 - NLP部分

***发展历史整体感知*** （从`综述`、博客不断积累，自己整理出整体发展的历史，再逐个掌握）

![img.png](pics/img.png)

```markdown 
-- https://markmap.js.org/repl
# NLP
## 经典模型

### 统计机器学习
#### ......

### 神经网络
#### 词向量
##### NNLM
##### Word2Vec
##### FaseText
##### Glove
#### CNN
#### RNN & LSTM & GRU
#### Bert 
#### ElMo?
```

综述1:
[2020 A Survey on Text Classification: From Shallow to Deep Learning](https://arxiv.org/pdf/2008.00364v2.pdf)

![img_1.png](pics/img_from-shallow-to-deep-learning-fig2.png)

## TextCNN

重要参考


| type 	| paper                                                                                                             	| intro                                                                         	| link                                  	|
|------	|-------------------------------------------------	|-------------------------------------------------	|----------------	|
| 原作 	| 2014-Convolutional Neural Networks for Sentence Classification                                                         	| (1) CNN-random-init <br> (2)CNN-static <br> (3)CNN-non-static <br> (4)CNN-multichannel 	| [link](https://aclanthology.org/D14-1181.pdf) 	|
| 解读 	| 2016-A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification 	|                                                                                        	| [link](https://arxiv.org/pdf/1510.03820.pdf)  	|
|      	|                                                                                                                        	|                                                                                        	|                                               	|



### 原理推导

- 构造词（句）的向量表示：文章中使用到了 word2vec 训练的词向量，

$x_{i} /in R^{k}$: k be the k-dimensional word vector corresponding to the i-th word in the sentence

- 在一定窗口长度下构造句子的部分向量

$x_{1:n}= x_{1} \oplus x_{2} \oplus . . . \oplus x_{n}$







---
---
---

# 通用
### optimization
```python
model.compile(optimizer='rmsprop',...)
```
- 参考
- [深度学习优化算法解析(Momentum, RMSProp, Adam)](https://blog.csdn.net/willduan1/article/details/78070086)
- [Intro to optimization in deep learning: Momentum, RMSProp and Adam](https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/#:~:text=While%20momentum%20accelerates%20our%20search,of%20both%20Momentum%20and%20RMSProp.)

### 激活函数
- [激活函数总结](https://zhuanlan.zhihu.com/p/73214810)
- [逻辑回归原理小结](https://www.cnblogs.com/pinard/p/6029432.html)
- [梯度下降（Gradient Descent）小结](https://www.cnblogs.com/pinard/p/5970503.html)

### 计算
- [[矩阵分析与应用].张贤达.扫描版](https://github.com/61--/weiyanmin/blob/master/BOOK/%5B%E7%9F%A9%E9%98%B5%E5%88%86%E6%9E%90%E4%B8%8E%E5%BA%94%E7%94%A8%5D.%E5%BC%A0%E8%B4%A4%E8%BE%BE.%E6%89%AB%E6%8F%8F%E7%89%88.pdf)
- [DNN为例的梯度计算](https://zhuanlan.zhihu.com/p/29815081)


# 按照模型划分

## DNN系列
[发展历史介绍](https://www.jiqizhixin.com/graph/technologies/f82b7976-b182-40fa-b7d8-a3aad9952937#:~:text=%E4%B9%8B%E5%BF%83Techopedia-,%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88DNN%EF%BC%89%E6%98%AF%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E4%B8%80%E7%A7%8D,%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83%E3%80%82)


## RNN & GRU & LSTM

- 参考
    - [深度学习之循环神经网络（RNN）](https://www.cnblogs.com/Luv-GEM/p/10703906.html)
    - [TensorFlow之RNN：堆叠RNN、LSTM、GRU及双向LSTM](https://www.cnblogs.com/Luv-GEM/p/10788849.html)
    - [循环神经网络之LSTM和GRU](https://www.cnblogs.com/Luv-GEM/p/10705967.html)

- LSTM
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


## Transformer

论文解读
- 精读论文
  - [Transformer-<Attention-is-all-you-need>](https://github.com/Iven2166/models-learning/blob/main/paper-reading/Transformer-%3CAttention-is-all-you-need%3E.md)
  
- 其他参考
  - [公众号-实现](https://mp.weixin.qq.com/s/cuYAt5B22CYtB3FBabjEIw)
  

**历史发展**

| 进程     | 论文    | 
| :------------- | :------------- | 
| 概念提出      | 在人工智能领域,注意力这一概念最早是在计算机视觉中提出，用来提取图像特征．[[Itti et al., 1998](https://www.cse.psu.edu/~rtc12/CSE597E/papers/Itti_etal98pami.pdf) ]提出了一种自下而上的注意力模型． 该模型通过提取局部的低级视觉特征，得到一些潜在的显著（salient）区域．      | 
| 图像分类      | 在神经网络中， [[Mnih et al., 2014](https://arxiv.org/pdf/1406.6247.pdf) ]在循环神经网络模型上使用了注意力机制来进行图像分类．     | 
| 机器翻译      | [[Bahdanau et al., 2014](https://arxiv.org/pdf/1409.0473.pdf) ]使用注意力机制在机器翻译任务上将翻译和对齐同时进行．     | 
| attention is all you need      | 目前， 注意力机制已经在语音识别、图像标题生成、阅读理解、文本分类、机器 翻译等多个任务上取得了很好的效果， 也变得越来越流行． 注意力机制的一个重 要应用是自注意力． 自注意力可以作为神经网络中的一层来使用， 有效地建模长 距离依赖问题 [[Attention is all you need, Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf) ]     | 


# 领域划分
## NLP

---

- 一些博客参考
  
  - [rumor - 如何系统性地学习NLP 自然语言处理？ - `关注里面对综述的引用`](https://www.zhihu.com/question/27529154/answer/1643865710)
    - [NLP快速入门路线及任务详解 - `其他综述`](https://mp.weixin.qq.com/s/zrziuFLRvbG8axG48QpFvg)
    - [深度学习文本分类｜模型&代码&技巧](https://mp.weixin.qq.com/s?__biz=MzAxMTk4NDkwNw==&mid=2247485854&idx=1&sn=040d51b0424bdee66f96d63d4ecfbe7e&chksm=9bb980faacce09ec069afa79c903b1e3a5c0d3679c41092e16b2fdd6949aa059883474d0c2af&token=1473678595&lang=zh_CN&scene=21#wechat_redirect)
  - [nlp-tutorial](https://github.com/graykode/nlp-tutorial)
  - [非常牛逼的nlp_course！！！](https://github.com/yandexdataschool/nlp_course)
  - [nlp-roadmap](https://github.com/graykode/nlp-roadmap)
  - [Transformers-大集合](https://github.com/huggingface/transformers)
  - [个人博客1](https://wmathor.com/index.php/archives/1399/)
  - [自然语言处理入门-一些名词和概念](http://www.fanyeong.com/2018/02/13/introduction_to_nlp/)
  - [Archive | Deep Learning for Natural Language Processing](https://machinelearningmastery.com/category/natural-language-processing/)
  - [NLP101-github](https://github.com/Huffon/NLP101)
  - [text-classification](https://github.com/zhengwsh/text-classification)
  - [meta-research](https://github.com/orgs/facebookresearch/repositories?q=&type=all&language=python&sort=)
  


### Word Embedding 词嵌入

概念上而言，它是指把一个维数为所有词的数量的高维空间嵌入到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量。
[wiki](https://zh.wikipedia.org/wiki/%E8%AF%8D%E5%B5%8C%E5%85%A5)

特点
- 高维度转低维度表示：单词个数 |V| (one-hot) 转化为指定维度
- 词汇之间能够表示相似度
- 向量的每一维没有含义

文本表示有哪些方法？
- 基于one-hot、tf-idf、textrank等的bag-of-words 
  - 维度灾难
  - 语义鸿沟
- 主题模型：LSA（SVD）、pLSA、LDA
  - 计算量复杂
- 基于词向量的固定表征：word2vec、fastText、glove
  - 相同上下文语境的词有似含义
  - 固定表征无法表示"一词多义"（因为一个单词只有一个emb？）
- 基于词向量的动态表征：elmo、GPT、bert

单个介绍
- word2vec(2013)
  - 论文：
    - 原作者-[word2vec思想](https://arxiv.org/pdf/1301.3781.pdf) (也讨论了和NNLM等的区别)
    - 原作者-Skip-gram模型的两个策略：[Hierarchical Softmax 和 Negative Sampling](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
    - 原作者的[博士论文（2012），适合用于了解历史](https://www.fit.vut.cz/study/phd-thesis-file/283/283.pdf)
    - 其他作者解读
      - [word2vec Parameter Learning Explained](https://arxiv.org/pdf/1411.2738.pdf)
      - [word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method](https://arxiv.org/pdf/1402.3722.pdf)
  - 要点
    - Huffman Tree 霍夫曼二叉树：权值更高的离树越近
  - 博客参考：
    - [word2vec原理(一) CBOW与Skip-Gram模型基础](https://www.cnblogs.com/pinard/p/7160330.html)
    - [word2vec原理(二) 基于Hierarchical Softmax的模型](https://www.cnblogs.com/pinard/p/7243513.html)
    
  - 其实word2vec和Co-Occurrence Vector的思想是很相似的，都是基于一个统计学上的假设：经常在同一个上下文出现的单词是相似的。只是他们的实现方式是不一样的，前者是采用词频统计，降维，矩阵分解等确定性技术；而后者则采用了神经网络进行不确定预测，它的提出主要是采用神经网络之后计算复杂度和最终效果都比之前的模型要好。所以那篇文章的标题才叫：Efficient Estimation of Word Representations in Vector Space。[参考](http://www.fanyeong.com/2017/10/10/word2vec/) 
- Glove
- fastText
- elmo

---
*Glove*


- 参考：
- [官网](https://nlp.stanford.edu/projects/glove/)
- [git](https://github.com/stanfordnlp/GloVe)
- [论文](https://nlp.stanford.edu/pubs/glove.pdf)

---

### 预训练


- [不错的博主](https://github.com/loujie0822/Pre-trained-Models)
  - [NLP算法面试必备！PTMs：NLP预训练模型的全面总结](https://zhuanlan.zhihu.com/p/115014536)
  - [nlp中的词向量对比：word2vec/glove/fastText/elmo/GPT/bert](https://zhuanlan.zhihu.com/p/56382372)
  - [nlp中的预训练语言模型总结(单向模型、BERT系列模型、XLNet)](https://zhuanlan.zhihu.com/p/76912493)
- [Glove详解](http://www.fanyeong.com/2018/02/19/glove-in-detail/#comment-1462)
- [神经网络语言模型(NNLM)](https://blog.csdn.net/u010089444/article/details/52624964)

- [NLP必读 | 十分钟读懂谷歌BERT模型](https://www.jianshu.com/p/4dbdb5ab959b)
- [Transformer-论文解读](https://www.jianshu.com/p/4b1bcd5c5f80)

- NNLM
  - 论文:  https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf 
  - 论文解读: https://www.jianshu.com/p/be242ed3f314
  - [!!! NNLM 的 PyTorch 实现](https://wmathor.com/index.php/archives/1442/)


# 其他参考
|名称|描述|link|
|:---|---|:---|
|互联网核心应用（搜索、推荐、广告）算法宝藏书.pdf|结合公司业务进行的顶层设计介绍（广泛）|[PDF](https://livehbsaas.oss-cn-beijing.aliyuncs.com/%E4%BA%92%E8%81%94%E7%BD%91%E6%A0%B8%E5%BF%83%E5%BA%94%E7%94%A8%EF%BC%88%E6%90%9C%E7%B4%A2%E3%80%81%E6%8E%A8%E8%8D%90%E3%80%81%E5%B9%BF%E5%91%8A%EF%BC%89%E7%AE%97%E6%B3%95%E5%AE%9D%E8%97%8F%E4%B9%A6.pdf)|
|知乎-视频合集|一些深度学习、机器学习的视频合集|[link](https://space.bilibili.com/73012391/video?tid=0&page=2&keyword=&order=pubdate)|




