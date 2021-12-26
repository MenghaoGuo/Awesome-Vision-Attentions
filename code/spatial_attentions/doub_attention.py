# A2-Nets: Double Attention Networks (NIPS 2018)
import jittor as jt
from jittor import nn


class DoubleAtten(nn.Module):
    def __init__(self, in_c):
        super(DoubleAtten, self).__init__()
        self.in_c = in_c
        self.convA = nn.Conv2d(in_c, in_c, kernel_size=1)
        self.convB = nn.Conv2d(in_c, in_c, kernel_size=1)
        self.convV = nn.Conv2d(in_c, in_c, kernel_size=1)

    def execute(self, input):

        feature_maps = self.convA(input)
        atten_map = self.convB(input)
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w)
        atten_map = atten_map.view(b, self.in_c, 1, h*w)
        global_descriptors = jt.mean(
            (feature_maps * nn.softmax(atten_map, dim=-1)), dim=-1)

        v = self.convV(input)
        atten_vectors = nn.softmax(
            v.view(b, self.in_c, h*w), dim=-1)
        out = nn.bmm(atten_vectors.permute(0, 2, 1),
                     global_descriptors).permute(0, 2, 1)

        return out.view(b, _, h, w)


def main():
    attention_block = DoubleAtten(64)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    jt.grad(output, input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
