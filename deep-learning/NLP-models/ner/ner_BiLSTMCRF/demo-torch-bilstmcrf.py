
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item() # tensor([0]) --> 0


tmp = torch.randn(1, 3)
# print(tmp)
# print(argmax(tmp))


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    """
    在前向算法里稳定地计算log-sum exp
    最后return处应用了计算技巧，目的是防止sum后数据过大越界，实际就是对vec应用log_sum_exp

    :param vec:
    :return:
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim):
        """
        参数定义
        :param vocab_size: 词汇表大小
        :param tag_to_idx: 标签以及对应的序号
        :param embedding_dim:
        :param hidden_dim:
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 定义lstm， hidden_dim 为何除以2？ num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True)

        # maps the output of the LSTM into tag space
        # 将LSTM的输出转换到tag对应空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # matrix of transition parameters. Entry i, j is the score of transitioning to i from j
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # These two statements enforce the constraint that we never transfer to the start tag
        # and we never transfer from the stop tag
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        因为双向所以hidden_dim除以2？
        :return: h0 and c0
        h0.shape: tensor of shape (D * num_layers, N, H_out)
        c0.shape: tensor of shape (D * num_layers, N, H_cell)
        D = 2 if bi else 1
        N = batch_size
        """
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        """
        do the forward algorithm to compute the partition function
        :param feats: 表示发射矩阵(emit score)，实际上就是LSTM的输出，
        意思是经过LSTM的sentence的每个word对应于每个label的得分
        :return:
        """
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score
        # 因为start tag是4，所以tensor([[-10000., -10000., -10000., 0., -10000.]])，
        init_alphas[0][self.tag_to_idx[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # print("forward_var", forward_var.shape) # torch.Size([1, 5])

        # Iterate through the sentence
        for feat in feats:
            alphas_t = [] # the forward tensors at this timestep # 当前时间步的正向tensor
            for next_tag in range(self.tagset_size):

                # broadcast the emission score: it is the same regardless of the previous tag
                # LSTM的生成矩阵是emit_score，维度为1*5
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)

                # The i_th entry of trans_score is the score of transitioning to next_tag from i
                # self.transitions[next_tag] 这一行是各个 tag 转移到 next_tag 的分
                trans_score = self.transitions[next_tag].view(1, -1)

                # The i_th entry of next_tag_var is the value for the edge (i -> next_tag)
                # before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score

                # The forward variable for this tag is log-sum-exp of all the scores
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                # 此时的alphas t 是一个长度为5，例如<class 'list'>:
                # [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        """
        input (embeds): shape (L, N, H_in) when batch_first = False
        hidden: h_0 and c_0
        lstm_out: shape of (L, N, D * H_out)

        L = sequence length
        N = batch size
        H_in = input size
        H_out = proj_size if proj_size > 0 else hidden_size

        :param sentence: 看起来是单个句子
        :return:
        """
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        """
        Gives the score of a provided tag sentence
        含有 tag 的句子的总体分数

        :param feats:
        :param tags:
        :return:
        """
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long),
                          tags]) # 给tags前面加上start
        for i, feat in enumerate(feats): # feats: (句子长度, tag_size)
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]] # tags[i] to tags[i+1], tags最前面有一个start
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """
        预测序列的得分，维特比解码，输出得分与路径值
        :param feats:
        :return:
        """
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.0)
        init_vvars[0][self.tag_to_idx[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = [] # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag
                # we don't include the emission scores here because the max does not depend on them
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG， 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_idx[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        """

        :param sentence:
        :param tags:
        :return:
        """
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags) # 完美的分数
        return forward_score - gold_score

    def forward(self, sentence):
        """
        Get the emission scores from the BiLSTM
        :param sentence:
        :return:
        """
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# 样例
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

training_data = [
    (
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ),
    (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )
]

word_to_idx = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

tag_to_idx = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
idx_to_tag = dict()
for i in tag_to_idx:
    idx_to_tag[tag_to_idx[i]] = i

model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_idx)
    precheck_tags = torch.tensor([tag_to_idx[t] for t in training_data[0][1]], dtype=torch.long)
    # print(model(precheck_sent))

for epoch in range(7):
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_idx)
        targets = torch.tensor([tag_to_idx[t] for t in tags], dtype=torch.long)
        # print(sentence_in, targets)
        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters
        # by calling optimizer.step()
        loss.backward()
        optimizer.step()
        print(loss)

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_idx)
    print(training_data[0][0])
    score, res = model(precheck_sent)
    print(score, res, [idx_to_tag[i] for i in list(res)])
