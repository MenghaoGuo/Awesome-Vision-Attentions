import jittor as jt
from jittor import nn


class GlobalContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio):
        super(GlobalContextBlock, self).__init__()
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(),  # yapf: disable
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = jt.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def execute(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term

        return out


def main():
    attention_block = GlobalContextBlock(64, 1/4)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
