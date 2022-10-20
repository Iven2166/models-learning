

（1）V0: Embrand200-bilstm1Layer200Hidden16Batch1e-3Learn

- token-emb：nn.embedding随机初始化， 200维度
- bilstm：200hidden、1layer、16batch-size、1e-3learning-rate

（2）V1:
基于V0改动了layer=3，发现效果并不好。

加入了两层FC，并且在中间进行dropout=0.2进行训练

epoch=50时，效果没有v0好

再增加20个epoch

（3）V2:
加入GLove的读取，作为emb的pretrained，效果提升很多

(4)V3:加入CRF层
首尾加入start和stop，所以需要在token原文、token_id的seq里加入、token_emb里加入。
token在首尾分别加入：START_TAG = "<START>"，STOP_TAG = "<STOP>"

不用lstm对start和stop的emb进行预测，只预测之间的；再补上两边的tag，用crf进行预测？
