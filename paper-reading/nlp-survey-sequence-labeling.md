
# 综述-序列标注
[ A Survey on Recent Advances in Sequence Labeling from Deep Learning Models](https://arxiv.org/pdf/2011.06727.pdf)

- 介绍深度学习下的序列标注 (Sequence Labeling, SL) 的三种任务：part-of-speech tagging (POS), named entity recognition (NER) and text chunking

1. Introduction 
    1. 1～2段：
下游任务：实体关系识别、
传统的方法：HMM、CRF，同时依赖于文本的手工特征（首字母是否大写、语言特定特征-地名等）

    2. 4段：Contributions of this survey 
将SL的领域分为三个来介绍：embedding module, context encoder module, inference module

2. Background 
    1. 任务介绍
        1. POS 标注词性
        2. NER 标注实体
            1. 应用业务广泛：搜索、问答、知识图谱、翻译（词组如果直接翻译，是单词翻译后的拼凑；而NER能够理解词组先后顺序、范围）
            1. 标注类型
                1. 三大类：entity, time, and numeric
                2. 七小类：person name, organization, location, time, date, currency, and percentage
                3. CoNLL2003 NER 制定：person (PER), location (LOC), organization (ORG) and proper nouns (MISC)
            2. 标注规则
                1. BIOES system： “B” (Begin), “I” (Inside), “E” (End), 
                   “0-” (Outside) means it does not belong to any named entity phrase
                   “S-” (Single) indicates it is the only word that represent an entity
        3. Text Chunking 句子分割句法上独立的、不重叠的词组
    2. 传统的机器学习方法
        1. 模型：HMM、SVM、Maximum Entropy Models、CRF
    3. 深度学习方法
        1. 分 module 介绍
            1. embedding encoder module: 词的emb映射
            2. context encoder module: 提取上下文特征
            3. inference module: 预测最佳概率的label
        2. embedding encoder module
            1. Pretrained Word Embeddings: word2vec, senna, bi-LSTM, ELMo, BERT,
            2. Character-level Representations: CNN, RNN, 用于建模 character to word embeddings; 有助于学习单词的字符表示
            3. Hand-crafted features
                1. word spelling features: word suffix, word capitalization, gazetteer 
                2. 从Wiki提取预识别过的120个实体类型的多种单词，和其他单词一起表示为低维的emb，再计算每个单词和120个命名实体类型词汇的关系，从而计算emb。
                   比如靠近实体的词汇能够学习关系，从而计算emb `Obama` had won the `2009 Nobel Peace Prize` [Robust Lexical Features, Ghaddar et al.](https://aclanthology.org/C18-1161.pdf)
        3. Context Encoder Module: 学习每个token的上下文信息、依赖关系
            1. RNN类
            2. CNN: 用于捕捉一定窗口里的token之间特征；其次，并行比RNN快。GCNN版本是加了gate。
                不足在于CNN难以捕捉长距离的依赖关系。
            3. Transformer
        4. Inference Module
            1. Softmax 
            2. CRFs: 比如softmax是单独地对token的label进行预测，而没有考虑label之间的依赖关系。
                而且label的正确选择往往取决于附近单词的信息。
                1. CRFs: 对单词进行预测和label之间关系的转移矩阵学习
                2. semi-CRFs: 对输入序列的segment (`subsequence`: 
                   [Semi-Markov Conditional Random Fields for Information Extraction](https://www.cs.cmu.edu/~wcohen/postscript/semiCRF.pdf) )
                   进行建模（而非单词维度）；缺点是增加了计算复杂度
                3. Hybrid Semi-CRFs(HSCRF): 同时对单词和segment进行建模
                4. NCRF transducers (neural CRF transducers): 利用
                





