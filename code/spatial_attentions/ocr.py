import jittor as jt 
import jittor.nn as nn
from jittor import init

class OCRHead(nn.Module):
    def __init__(self, in_channels, n_cls=19):
        super(OCRHead, self).__init__() 
        self.relu = nn.ReLU() 
        self.in_channels = in_channels
        self.softmax = nn.Softmax(dim = 2)
        self.conv_1x1 = nn.Conv(in_channels, in_channels, kernel_size=1)
        self.last_conv = nn.Conv(in_channels * 2, n_cls, kernel_size=3, stride=1, padding=1)
        self._zero_init_conv() 
    def _zero_init_conv(self):
        self.conv_1x1.weight = init.constant([self.in_channels, self.in_channels, 1, 1], 'float', value=0.0)

    def execute(self, context, feature):
        batch_size, c, h, w = feature.shape 
        origin_feature = feature 
        feature = feature.reshape(batch_size, c, -1).transpose(0, 2, 1) # b, h*w, c
        context = context.reshape(batch_size, context.shape[1], -1) # b, n_cls, h*w
        attention = self.softmax(context)
        ocr_context = nn.bmm(attention, feature).transpose(0, 2, 1) # b, c, n_cls
        relation = nn.bmm(feature, ocr_context).transpose(0, 2, 1) # b, n_cls, h*w
        attention = self.softmax(relation) #b , n_cls, h*w 
        result = nn.bmm(ocr_context, attention).reshape(batch_size, c, h, w) 
        result = self.conv_1x1(result)
        result = jt.concat ([result, origin_feature], dim=1)
        result = self.last_conv (result)
        return result




