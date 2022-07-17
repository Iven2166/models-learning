

参考：https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html



lstm函数用于 return (句子长度, lstm对于tag类别的概率值判断 )

```python
def _get_lstm_features(self, sentence):
    self.hidden = self.init_hidden()
    embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
    lstm_out, self.hidden = self.lstm(embeds, self.hidden)
    lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
    lstm_feats = self.hidden2tag(lstm_out)
    return lstm_feats
```

