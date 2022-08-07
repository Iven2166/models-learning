
论文综述：[A Survey on Deep Learning for Named Entity Recognition](https://arxiv.org/pdf/1812.09449.pdf)

摘要：命名实体识别 (NER) 的任务是从属于预定义语义类型（如人员、位置、组织等）的文本中识别刚性指示符的提及。NER 始终作为许多自然语言应用程序的基础，如问答、文本摘要、和机器翻译。早期的 NER 系统在设计特定领域的特征和规则时以人力工程成本实现了良好的性能，取得了巨大的成功。近年来，通过非线性处理，连续实值向量表示和语义组合赋予深度学习能力，已在 NER 系统中得到应用，产生了最先进的性能。在本文中，我们对现有的 NER 深度学习技术进行了全面回顾。我们首先介绍 NER 资源，包括标记的 NER 语料库和现成的 NER 工具。然后，我们基于三个轴的分类系统对现有作品进行分类：输入的分布式表示、上下文编码器和标签解码器。接下来，我们调查了最近在新的 NER 问题设置和应用中应用深度学习技术的最具代表性的方法。最后，我们向读者介绍了 NER 系统面临的挑战，并概述了该领域的未来方向。
>Named entity recognition (NER) is the task to identify mentions of rigid designators from text belonging to predefined semantic types such as person, location, organization etc. NER always serves as the foundation for many natural language applications such as question answering, text summarization, and machine translation. Early NER systems got a huge success in achieving good performance with the cost of human engineering in designing domain-specific features and rules. In recent years, deep learning, empowered by continuous real-valued vector representations and semantic composition through nonlinear processing, has been employed in NER systems, yielding stat-of-the-art performance. In this paper, we provide a comprehensive review on existing deep learning techniques for NER. We first introduce NER resources, including tagged NER corpora and off-the-shelf NER tools. Then, we systematically categorize existing works based on a taxonomy along three axes: distributed representations for input, context encoder, and tag decoder. Next, we survey the most representative methods for recent applied techniques of deep learning in new NER problem settings and applications. Finally, we present readers with the challenges faced by NER systems and outline future directions in this area.

进行序列标注时，单词的部分（ intra-word 前缀、后缀）非常重要，因此有 character-level representation 的做法