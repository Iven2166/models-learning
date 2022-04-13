Efficient-Estimation-of-Word-Representations-in-Vector-Space
---

paper: https://arxiv.org/pdf/1301.3781.pdf

---
目录
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


