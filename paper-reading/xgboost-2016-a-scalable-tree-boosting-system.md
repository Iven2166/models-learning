
https://arxiv.org/pdf/1603.02754.pdf

摘要
> Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.
>
> 树提升是一种高效且广泛使用的机器学习方法。 在本文中，我们描述了一种名为 XGBoost 的可扩展端到端树增强系统，该系统被数据科学家广泛用于在许多机器学习挑战中实现最先进的结果。 
> 
> 我们提出了一种新颖的稀疏数据感知算法和用于近似树学习的加权分位数草图。 更重要的是，我们提供有关缓存访问模式、数据压缩和分片的见解，以构建可扩展的树增强系统。 通过结合这些见解，XGBoost 使用比现有系统少得多的资源扩展了数十亿个示例。

# 1. Introduction

介绍了梯度提升树在应用里的优势，用于排名预测、点击率预测、集成方法。
> Among the machine learning methods used in practice, gradient tree boosting [10] 1 is one technique that shines in many applications. Tree boosting has been shown to give state-of-the-art results on many standard classification benchmarks [16]. LambdaMART [5], a variant of tree boosting for ranking, achieves state-of-the-art result for ranking problems. Besides being used as a stand-alone predictor, it is also incorporated into real-world production pipelines for ad click through rate prediction [15]. Finally, it is the defacto choice of ensemble method and is used in challenges such as the Netflix prize [3].
>
> 在实践中使用的机器学习方法中，梯度树提升 [10] 1 是一种在许多应用中大放异彩的技术。 树增强已被证明可以在许多标准分类基准上提供最先进的结果 [16]。 LambdaMART [5] 是一种用于排名的树提升的变体，在排名问题上取得了最先进的结果。 除了用作独立的预测器外，它还被整合到现实世界的生产管道中，用于广告点击率预测 [15]。 最后，它是集成方法的实际选择，并用于 Netflix 奖 [3] 等挑战。

xgboost在kaggle的成绩 & 实际的一些应用（商店销售预测； 高能物理事件分类； 网页文本分类； 客户行为预测； 运动检测; 广告点击率预测； 恶意软件分类； 产品分类； 危害风险预测； 海量在线课程辍学率预测）
> In this paper, we describe XGBoost, a scalable machine learning system for tree boosting. The system is available as an open source package2 . The impact of the system has been widely recognized in a number of machine learning and data mining challenges. Take the challenges hosted by the machine learning competition site Kaggle for example. Among the 29 challenge winning solutions 3 published at Kaggle’s blog during 2015, 17 solutions used XGBoost. Among these solutions, eight solely used XGBoost to train the model, while most others combined XGBoost with neural nets in ensembles. For comparison, the second most popular method, deep neural nets, was used in 11 solutions. The success of the system was also witnessed in KDDCup 2015, where XGBoost was used by every winning team in the top-10. Moreover, the winning teams reported that ensemble methods outperform a well-configured XGBoost by only a small amount [1].
> 
> These results demonstrate that our system gives state-ofthe-art results on a wide range of problems. Examples of the problems in these winning solutions include: store sales prediction; high energy physics event classification; web text classification; customer behavior prediction; motion detection; ad click through rate prediction; malware classification; product categorization; hazard risk prediction; massive online course dropout rate prediction. While domain dependent data analysis and feature engineering play an important role in these solutions, the fact that XGBoost is the consensus choice of learner shows the impact and importance of our system and tree boosting.

- 高度可扩展的端到端的树增强系统
- 理论上合理的加权分位数草图进行高效计算
- 在并行树学习里引入稀疏感知算法
- 缓存感知块结构以进行核外树学习计算

> The major contributions of this paper:
> - We design and build a highly scalable end-to-end tree boosting system.
>- We propose a theoretically justified weighted quantile sketch for efficient proposal calculation.
>- We introduce a novel sparsity-aware algorithm for parallel tree learning.
>- We propose an effective cache-aware block structure for out-of-core tree learning

进一步强调了和现存算法研究的gap——现存算法虽然有一些现有的关于并行树提升的工作 [22、23、19]，但尚未探索诸如`核外计算、缓存感知和稀疏感知学习等方向`。
> While there are some existing works on parallel tree boosting [22, 23, 19], `the directions such as out-of-core computation, cache-aware and sparsity-aware learning have not been explored`. More importantly, an end-to-end system that combines all of these aspects gives a novel solution for real-world use-cases. This enables data scientists as well as researchers to build powerful variants of tree boosting algorithms [7, 8]. Besides these major contributions, we also make additional improvements in proposing a regularized learning objective, which we will include for completeness.

余文结构 (section)
2. 回顾树提升方法，并且引入正则化目标
3. 分裂发现方法
4. 系统设计，包括了相关实验结果，为每一个优化提供定量支持
5. 相关研究
6. 端到端的评估
7. 结论
> The remainder of the paper is organized as follows. We will first review tree boosting and introduce a regularized objective in Sec. 2. We then describe the split finding methods in Sec. 3 as well as the system design in Sec. 4, including experimental results when relevant to provide quantitative support for each optimization we describe. Related work is discussed in Sec. 5. Detailed end-to-end evaluations are included in Sec. 6. Finally we conclude the paper in Sec. 7.

# 2. TREE BOOSTING IN A NUTSHELL

# 2.1 Regularized Learning Objective








# 7. Conclusion
- 可扩展的树增强系统
- 我们提出了一种新颖的稀疏感知算法来处理稀疏数据和理论上合理的加权分位数草图用于近似学习。
- 缓存访问模式、数据压缩和分片是构建用于树提升的可扩展端到端系统的基本要素

> In this paper, we described the lessons we learnt when building XGBoost, a scalable tree boosting system that is widely used by data scientists and provides state-of-the-art results on many problems. We proposed a novel sparsity aware algorithm for handling sparse data and a theoretically justified weighted quantile sketch for approximate learning. Our experience shows that cache access patterns, data compression and sharding are essential elements for building a scalable end-to-end system for tree boosting. These lessons can be applied to other machine learning systems as well. By combining these insights, XGBoost is able to solve realworld scale problems using a minimal amount of resources.



