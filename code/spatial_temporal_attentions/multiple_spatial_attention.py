# Diversity regularized spatiotemporal attention for video-based person re-identification (arXiv 2018)
import jittor as jt
from jittor import nn


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def execute(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).execute(input)
        size = input.size()[:2]
        out = super(Bottle, self).execute(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass


class sattention(nn.Module):
    def __init__(self, input_channel, output_channel=128, seqlen=6, norm=True, spanum=3):
        super(sattention, self).__init__()

        self.atn_height = 8
        self.atn_width = 4
        self.spanum = spanum
        self.seqlen = seqlen
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.norm = norm

        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            input_channel, self.spanum, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.spanum)

        self.softmax = BottleSoftmax()

        self.feat = nn.Linear(input_channel, self.output_channel)

    def execute(self, x):

        atn = x
        atn = self.conv1(atn)
        atn = self.bn1(atn)
        atn = self.relu1(atn)
        atn = self.conv2(atn)
        atn = self.bn2(atn)
        atn = atn.view(-1, self.spanum, self.atn_height*self.atn_width)
        atn = self.softmax(atn)

        # Diversity Regularization
        reg = atn

        # Multiple Spatial Attention
        atn = atn.view(atn.size(0), self.spanum, 1,
                       self.atn_height, self.atn_width)
        atn = atn.expand(atn.size(0), self.spanum, self.input_channel,
                         self.atn_height, self.atn_width)
        x = x.view(x.size(0), 1, self.input_channel,
                   self.atn_height, self.atn_width)
        x = x.expand(x.size(0), self.spanum, self.input_channel,
                     self.atn_height, self.atn_width)

        x = x * atn
        x = x.view(-1, self.input_channel, self.atn_height, self.atn_width)
        x = nn.avg_pool2d(x, x.size()[2:])*x.size(2)*x.size(3)
        x = x.view(-1, self.input_channel)

        x = self.feat(x)
        x = x.view(-1, self.spanum, self.num_features)

        if self.norm:
            x = x / x.norm(2, 2).view(-1, self.spanum, 1).expand_as(x)
        x = x.view(-1, self.seqlen, self.spanum, self.num_features)

        return x, reg


def main():
    attention_block = sattention(64, 64)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
