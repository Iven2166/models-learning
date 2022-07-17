

### 模型架构

![img.png](img.png)

1. 字词嵌入 作为输入
2. BiLSTM输出标签预测
3. 输入CRF进一步学习各标签转移概率，整句的最佳概率，选择预测得分最高的标签序列作为最佳答案


# CRF 的作用

CRF层可以向最终的预测标签添加一些约束，以确保它们是有效的。这些约束可以由CRF层在训练过程中从训练数据集自动学习。

约束条件可以是：
- 句子中第一个单词的标签应该以“B-”或“O”开头，而不是“I-”
- “B-label1 I-label2 I-label3 I-…”，在这个模式中，label1、label2、label3…应该是相同的命名实体标签。例如，“B-Person I-Person”是有效的，但是“B-Person I-Organization”是无效的。
- “O I-label”无效。一个命名实体的第一个标签应该以“B-”而不是“I-”开头，换句话说，有效的模式应该是“O B-label”

## CRF：Emission得分
这些emission分数来自BiLSTM层

## CRF：Transition得分
各个标签之间的所有得分（理解为转移概率？），该矩阵(T * T, T: tag_size)是BiLSTM-CRF模型的参数

## 实际路径得分

$$ P_total = P1 + P2 + ... + Pn = e^ {S1} + e^ {S2} + ... + e^ {Sn}$$


### 参考
- 代码应用：https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
- [通俗易懂！BiLSTM上的CRF，用命名实体识别任务来解释CRF（一）](https://zhuanlan.zhihu.com/p/119254570)
