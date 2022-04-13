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



# 6. Conclusion

![img.png](img.png)
- 本文中的主要工作：在一系列的质量句法和语义语言任务(syntactic and semantic language tasks)上，研究了不同模型下的词向量表示的质量。

- 模型架构相对简单，提升计算速度，从而纳入了更多的训练样本。
    - 由于计算复杂度低得多，可以从更大的数据集中计算出非常准确的高维词向量
    - 与NNLM、RNNLM对比（文中的第2章）
    
- 应用DistBelief 分布式框架：训练 CBOW 和 Skip-gram，即使在具有一万亿个单词的语料库上也可以建模，基本上是无限大小的词汇表




