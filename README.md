*models-learning*

本文件夹用于积累对模型的学习及定期回顾。

- 整体的版图认知
- 各个模型的认知 & 实践 （一个模型一个文件夹，需要包括md 及 ipynb模型训练文件）

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

# 按照模型划分
## RNN & GRU & LSTM

- 参考
    - [深度学习之循环神经网络（RNN）](https://www.cnblogs.com/Luv-GEM/p/10703906.html)
    - [TensorFlow之RNN：堆叠RNN、LSTM、GRU及双向LSTM](https://www.cnblogs.com/Luv-GEM/p/10788849.html)
    - [循环神经网络之LSTM和GRU](https://www.cnblogs.com/Luv-GEM/p/10705967.html)


# 领域划分
## NLP

### Word Embedding 词嵌入

概念上而言，它是指把一个维数为所有词的数量的高维空间嵌入到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量。
[wiki](https://zh.wikipedia.org/wiki/%E8%AF%8D%E5%B5%8C%E5%85%A5)

特点
- 高维度转低维度表示：单词个数 |V| (one-hot) 转化为指定维度
- 词汇之间能够表示相似度
- 向量的每一维没有含义

文本表示有哪些方法？
- 基于one-hot、tf-idf、textrank等的bag-of-words 
  - 维度灾难、语义鸿沟
- 主题模型：LSA（SVD）、pLSA、LDA
  - 计算量复杂
- 基于词向量的固定表征：word2vec、fastText、glove
  - 相同上下文语境的词有似含义
  - 固定表征无法表示"一词多义"（因为一个单词只有一个emb）
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
- 博客参考
- [nlp-tutorial](https://github.com/graykode/nlp-tutorial)
- [nlp-roadmap](https://github.com/graykode/nlp-roadmap)
- [Transformers-大集合](https://github.com/huggingface/transformers)
- [个人博客1](https://wmathor.com/index.php/archives/1399/)
- [自然语言处理入门-一些名词和概念](http://www.fanyeong.com/2018/02/13/introduction_to_nlp/)
- [Archive | Deep Learning for Natural Language Processing](https://machinelearningmastery.com/category/natural-language-processing/)




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
  - 论文: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
  - 论文解读: https://www.jianshu.com/p/be242ed3f314
  - [!!! NNLM 的 PyTorch 实现](https://wmathor.com/index.php/archives/1442/)





