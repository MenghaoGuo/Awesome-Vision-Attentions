import jittor as jt
import jittor.nn as nn


class Highway(nn.Module):
    def __init__(self, dim, num_layers=2):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                  for _ in range(num_layers)])

        self.f = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = self.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
            print(x.size())
        return x


def main():
    attention_block = Highway(32)
    input = jt.rand([4, 64, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
