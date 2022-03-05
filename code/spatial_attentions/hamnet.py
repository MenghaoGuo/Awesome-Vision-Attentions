# Is Attention Better Than Matrix Decomposition? (ICLR 2021)
import jittor as jt
from jittor import nn
from contextlib import contextmanager


@contextmanager
def null_context():
    yield


class NMF(nn.Module):
    def __init__(
        self,
        dim,
        n,
        ratio=8,
        K=6,
        eps=2e-8
    ):
        super().__init__()
        r = dim // ratio

        self.D = jt.zeros((dim, r)).uniform_(0, 1)
        self.C = jt.zeros((r, n)).uniform_(0, 1)

        self.K = K

        self.eps = eps

    def execute(self, x):
        b, D, C, eps = x.shape[0], self.D, self.C, self.eps

        # x is made non-negative with relu as proposed in paper
        x = nn.relu(x)
        D = D.unsqueeze(0).repeat(b, 1, 1)
        C = C.unsqueeze(0).repeat(b, 1, 1)

        # transpose
        def t(tensor): return tensor.transpose(1, 2)

        for k in reversed(range(self.K)):
            # only calculate gradients on the last step, per propose 'One-step Gradient'
            context = null_context if k == 0 else jt.no_grad
            with context():
                C_new = C * ((t(D) @ x) / ((t(D) @ D @ C) + eps))
                D_new = D * ((x @ t(C)) / ((D @ C @ t(C)) + eps))
                C, D = C_new, D_new

        return D @ C


class Hamburger(nn.Module):
    def __init__(
        self,
        dim,
        n,
        inner_dim,
        ratio=8,
        K=6
    ):
        super().__init__()

        self.lower_bread = nn.Conv1d(dim, inner_dim, 1, bias=False)
        self.ham = NMF(inner_dim, n, ratio=ratio, K=K)
        self.upper_bread = nn.Conv1d(inner_dim, dim, 1, bias=False)

    def execute(self, x):
        input = x
        shape = x.shape
        x = x.flatten(2)

        x = self.lower_bread(x)
        x = self.ham(x)
        x = self.upper_bread(x)
        return input + x.reshape(shape)


def main():
    attention_block = Hamburger(64, 32*32, 64, 8, 6)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
