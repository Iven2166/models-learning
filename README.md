***models-learning***

该项目下汇总接触到的统计机器学习和深度学习模型，按照大任务可分为NLP、CTR等。

每个模型包括了理论梳理（模型概念、算法）以及小型项目实现。

同时会包括结合论文阅读、公开的业务分享来加深理解。






# 附录

网址 / 博客参考：

- https://data-science-blog.com/


---





- 整体的版图认知
- 各个模型的认知 & 实践 （一个模型一个文件夹，需要包括md 及 ipynb模型训练文件）

# 重构V1

# 统计学习方法

框架基于《统计学习方法-李航》，对经典的机器学习方法进行原理学习、复现。

## 集成学习


Hoeffding 不等式

重点参考
- 西瓜书 & 南瓜书 - 机器学习周志华：第8章 集成学习

## xgboost

原作：[2016 XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf)
> Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.
> 
> 树提升是一种高效且广泛使用的机器学习方法。 在本文中，我们描述了一种名为 XGBoost 的可扩展端到端树增强系统，该系统被数据科学家广泛用于在许多机器学习挑战中实现最先进的结果。 我们提出了一种新颖的稀疏数据感知算法和用于近似树学习的加权分位数草图。 更重要的是，我们提供有关缓存访问模式、数据压缩和分片的见解，以构建可扩展的树增强系统。 通过结合这些见解，XGBoost 使用比现有系统少得多的资源扩展了数十亿个示例。

扩展记录：
- xgboost官方文档
  - https://xgboost.readthedocs.io/en/stable/tutorials/model.html
  - [Frequently Asked Questions](https://xgboost.readthedocs.io/en/stable/faq.html)
  - [Notes on Parameter Tuning](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
  - [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- 原作论文阅读记录 [link](/paper-reading/xgboost-2016-a-scalable-tree-boosting-system.md)
- 书
  - 机器学习-周志华(西瓜书)结合南瓜书看解析 - 第8章 集成学习
  - 《李航-统计机器学习》-第8章 提升方法
- 知乎：
  - [【机器学习】决策树（下）——XGBoost、LightGBM（非常详细）](https://zhuanlan.zhihu.com/p/87885678)
  - [左手论文 右手代码 深入理解网红算法XGBoost](https://zhuanlan.zhihu.com/p/91817667)
  - [一篇文章搞定GBDT、Xgboost和LightGBM的面试](https://zhuanlan.zhihu.com/p/148050748)


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
- [wx-同样都是调参，为什么人家的神经网络比我牛逼100倍？](https://mp.weixin.qq.com/s/lbXUBcovfsL34tvgZkBZbw)
### 激活函数
- [激活函数总结](https://zhuanlan.zhihu.com/p/73214810)
- [逻辑回归原理小结](https://www.cnblogs.com/pinard/p/6029432.html)
- [梯度下降（Gradient Descent）小结](https://www.cnblogs.com/pinard/p/5970503.html)

### 计算
- [[矩阵分析与应用].张贤达.扫描版](https://github.com/61--/weiyanmin/blob/master/BOOK/%5B%E7%9F%A9%E9%98%B5%E5%88%86%E6%9E%90%E4%B8%8E%E5%BA%94%E7%94%A8%5D.%E5%BC%A0%E8%B4%A4%E8%BE%BE.%E6%89%AB%E6%8F%8F%E7%89%88.pdf)
- [DNN为例的梯度计算](https://zhuanlan.zhihu.com/p/29815081)
- [wiki-矩阵和特征向量、特征值](https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E5%80%BC%E5%92%8C%E7%89%B9%E5%BE%81%E5%90%91%E9%87%8F)
- [统计学-英文书](http://egrcc.github.io/docs/math/all-of-statistics.pdf)




# 按照模型划分

## DNN系列
[发展历史介绍](https://www.jiqizhixin.com/graph/technologies/f82b7976-b182-40fa-b7d8-a3aad9952937#:~:text=%E4%B9%8B%E5%BF%83Techopedia-,%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88DNN%EF%BC%89%E6%98%AF%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E4%B8%80%E7%A7%8D,%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83%E3%80%82)




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

- 学术论文 / 学会
  - [ACL-Annual Meeting of the Association for Computational Linguistics `选择ACL为语言模型`](https://aclanthology.org/)
  - [EMNLP - 2020](https://2020.emnlp.org/papers/main)

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

## NER识别

- [知乎-BiLSTM上的CRF，用命名实体识别任务来解释CRF（一）](https://zhuanlan.zhihu.com/p/119254570)
- [wx-缺少训练样本怎么做实体识别？小样本下的NER解决方法汇总](https://mp.weixin.qq.com/s/FH1cWxXlTFt0RdEipSJH1w)

# 其他参考
|名称|描述|link|
|:---|---|:---|
|互联网核心应用（搜索、推荐、广告）算法宝藏书.pdf|结合公司业务进行的顶层设计介绍（广泛）|[PDF](https://livehbsaas.oss-cn-beijing.aliyuncs.com/%E4%BA%92%E8%81%94%E7%BD%91%E6%A0%B8%E5%BF%83%E5%BA%94%E7%94%A8%EF%BC%88%E6%90%9C%E7%B4%A2%E3%80%81%E6%8E%A8%E8%8D%90%E3%80%81%E5%B9%BF%E5%91%8A%EF%BC%89%E7%AE%97%E6%B3%95%E5%AE%9D%E8%97%8F%E4%B9%A6.pdf)|
|知乎-视频合集|一些深度学习、机器学习的视频合集|[link](https://space.bilibili.com/73012391/video?tid=0&page=2&keyword=&order=pubdate)|

- 感觉整体整理的目录不错：https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/#tableofcontents
- NLP岗位八股文：https://zhuanlan.zhihu.com/p/470674031


---
面经
- [NLP算法](https://cloud.tencent.com/developer/article/1817838)
