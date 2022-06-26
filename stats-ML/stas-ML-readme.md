框架基于《统计学习方法-李航》，对经典的机器学习方法进行原理学习、复现。

重点参考
- 西瓜书 & 南瓜书 - 机器学习周志华：第8章 集成学习

## xgboost

原作：[2016 XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf)
> Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.
> 
> 树提升是一种高效且广泛使用的机器学习方法。 在本文中，我们描述了一种名为 XGBoost 的可扩展端到端树增强系统，该系统被数据科学家广泛用于在许多机器学习挑战中实现最先进的结果。 我们提出了一种新颖的稀疏数据感知算法和用于近似树学习的加权分位数草图。 更重要的是，我们提供有关缓存访问模式、数据压缩和分片的见解，以构建可扩展的树增强系统。 通过结合这些见解，XGBoost 使用比现有系统少得多的资源扩展了数十亿个示例。

重要概念：Hoeffding 不等式

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
