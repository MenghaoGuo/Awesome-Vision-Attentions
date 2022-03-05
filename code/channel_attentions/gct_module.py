# Gated channel transformation for visual recognition (CVPR2020)
import jittor as jt
from jittor import nn


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = jt.ones((1, num_channels, 1, 1))
        self.gamma = jt.zeros((1, num_channels, 1, 1))
        self.beta = jt.zeros((1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def execute(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum(2, keepdims=True).sum(3, keepdims=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdims=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = jt.abs(x)
            else:
                _x = x
            embedding = _x.sum(2, keepdims=True).sum(
                3, keepdims=True) * self.alpha
            norm = self.gamma / \
                (jt.abs(embedding).mean(dim=1, keepdims=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + jt.tanh(embedding * norm + self.beta)

        return x * gate


def main():
    attention_block = GCT(64)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
