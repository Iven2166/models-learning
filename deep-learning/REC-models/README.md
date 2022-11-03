# Recommender System 

## 模型 & 模型实现

### FM

- 重点：在于FM模型能够用隐向量来代表特征之间交互时的权重（内积）。公式经过变形，达到降低计算复杂度的目的。

【如何简化计算FM复杂度】：https://zhuanlan.zhihu.com/p/58160982

- 模型结构实现: [py-file](https://github.com/Iven2166/models-learning/blob/main/deep-learning/REC-models/FM/FM_module.py)

### deepFM

- 模型重点：wide&deep里面的wide更换为FM，意在于用FM的隐向量来直接代表deep模块的输入，从而让deep一开始就学到了FM学习到的特征。

- 复现：
  - [deepFM复现](https://github.com/Iven2166/models-learning/blob/main/deep-learning/REC-models/deepFM/deepFM-criteoSmall.ipynb) 
使用criteo-Small的点击数据，来构造模型预测CTR。 
  - 结果：测试集的AUC应该是在0.7-0.75左右，略低于公开记录。
  
### 多任务：MMOE

- 模型重点：第一个M是multi-gate，第二个M是multi-expert。由多个expert来生成向量，并且经过T个gate（scalar，广播的权重），
  分别作为T个tower的输入，最终产出概率。 （T是任务数量）本质是融合多个专家模型的意见，每个模型可能会有所倾向。
  
- 复现
  - [MMOE复现](https://github.com/Iven2166/models-learning/blob/main/deep-learning/REC-models/MMOE/%E5%A4%9A%E4%BB%BB%E5%8A%A1%E7%9B%AE%E6%A0%87%E5%AD%A6%E4%B9%A0-mmoe.ipynb)
使用公开数据集census-income来预测（1）收入是否破50k，以及（2）是否未婚
  - 结果：最终test task1 auc: 0.946, test task2 auc: 0.974。接近论文AUC

### 注意力：DIN
[doing]

# 参考
- d2l: https://d2l.ai/chapter_recommender-systems.html


