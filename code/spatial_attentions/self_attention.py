import jittor as jt
import jittor.nn as nn


class SelfAttention(nn.Module):
    """ self attention module"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query = nn.Conv(in_channels=in_dim,
                             out_channels=in_dim, kernel_size=1)
        self.key = nn.Conv(in_channels=in_dim,
                           out_channels=in_dim, kernel_size=1)
        self.value = nn.Conv(in_channels=in_dim,
                             out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query(x).reshape(
            m_batchsize, -1, width*height).transpose(0, 2, 1)
        proj_key = self.key(x).reshape(m_batchsize, -1, width*height)
        energy = nn.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).reshape(m_batchsize, -1, width*height)

        out = nn.bmm(proj_value, attention.transpose(0, 2, 1))
        out = out.reshape(m_batchsize, C, height, width)

        return out


def main():
    attention_block = SelfAttention(64)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
