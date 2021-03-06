
参考

- Bert论文：https://arxiv.org/abs/1810.04805
- [知乎1](https://zhuanlan.zhihu.com/p/426184475)
- [limu](https://www.bilibili.com/video/BV1PL411M7eQ?spm_id_from=333.999.0.0&vd_source=4089d4a51ca3637483befeb898ed1a46)


# 摘要
> We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
>
> 我们引入了一种称为 BERT 的新语言表示模型，它代表来自 Transformers 的双向编码器表示。
> 与最近的语言表示模型 (Peters et al., 2018a; Radford et al., 2018) 不同，BERT 旨在通过联合调节所有层的左右上下文来预训练来自未标记文本的深度双向表示。
> 因此，预训练的 BERT 模型可以通过一个额外的输出层进行微调，从而为各种任务（例如问答和语言推理）创建最先进的模型，而无需对特定于任务的架构进行大量修改。 
> BERT 在概念上很简单，在经验上很强大。它在 11 个自然语言处理任务上获得了新的最先进的结果，包括将 GLUE 分数推到 80.5%（7.7% 点的绝对改进），
> MultiNLI 准确度到 86.7%（4.6% 的绝对改进），SQuAD v1.1问答测试 F1 到 93.2（1.5 分绝对提高）和 SQuAD v2.0 测试 F1 到 83.1（5.1 分绝对提高）。

跟GPT单向（由左到右预测）的不同在于，bert运用双向。
ELMo是双向的RNN架构，但bert运用transformer

# introduction

在ELMo和GPT都是利用左到右的架构，来避免读到之后的信息。但实际运用上，例如句子的分类判断，问答其实
在从左到右和从右到左都是合法的；作者想到如果能够运用双向的信息应该可以提升效果。

带掩码的语言模型

分词：wordpiece，获取低频率词的词根
