# Simam: A simple, parameter-free attention module for convolutional neural networks (ICML 2021)
import jittor as jt
from jittor import nn


class simam_module(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def execute(self, x):

        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (
            x - x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)).pow(2)
        y = x_minus_mu_square / \
            (4 * (x_minus_mu_square.sum(dim=2,
             keepdims=True).sum(dim=3,
             keepdims=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


def main():
    attention_block = simam_module()
    input = jt.ones([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
