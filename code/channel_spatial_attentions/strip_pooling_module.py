# Strip Pooling: Rethinking spatial pooling for scene parsing (CVPR 2020)
import jittor as jt
from jittor import nn


class StripPooling(nn.Module):
    """
    Reference:
    """

    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                   norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def execute(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = nn.interpolate(self.conv2_1(self.pool1(x1)),
                              (h, w), **self._up_kwargs)
        x2_3 = nn.interpolate(self.conv2_2(self.pool2(x1)),
                              (h, w), **self._up_kwargs)
        x2_4 = nn.interpolate(self.conv2_3(self.pool3(x2)),
                              (h, w), **self._up_kwargs)
        x2_5 = nn.interpolate(self.conv2_4(self.pool4(x2)),
                              (h, w), **self._up_kwargs)
        x1 = self.conv2_5(nn.relu(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(nn.relu(x2_5 + x2_4))
        out = self.conv3(jt.concat([x1, x2], dim=1))
        return nn.relu(x + out)


def main():
    attention_block = StripPooling(
        64, (20, 12), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
