import torch
from torch import nn


class FM(nn.Module):

    def __init__(self, latent_dim, feature_dim):
        """

        :param latent_dim: FM 的v，作为交互的隐向量
        :param feature_dim: 特征维度 x [batch_size, feature_num]
        """
        super(FM, self).__init__()

        self.latent_dim = latent_dim
        self.feature_dim = feature_dim

        self.w0 = nn.Parameter(torch.rand(1,))
        self.w1 = nn.Parameter(torch.rand([self.feature_dim, 1]))
        self.w2 = nn.Parameter(torch.rand([self.feature_dim, self.latent_dim]))

    def forward(self, inputs):
        first_order = self.w0 + torch.mm(inputs, self.w1) # shape = [batch_size, 1]
        second_order = 1 / 2 * torch.sum(
            torch.pow(torch.mm(inputs, self.w2), 2) - torch.mm(torch.pow(inputs, 2), torch.pow(self.w2, 2)),
            dim=1,
            keepdim=True
        ) # shape = [batch_size, 1]
        return first_order + second_order


if __name__=='__main__':

    batch_size = 100
    hidden_dim = 20
    feature_dim = 60
    inputs = torch.rand([batch_size, feature_dim])
    fm = FM(latent_dim=hidden_dim, feature_dim=feature_dim)
    b = fm(inputs)
    print(b, b.size())