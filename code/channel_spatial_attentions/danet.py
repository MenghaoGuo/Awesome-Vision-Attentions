import jittor as jt
import jittor.nn as nn


class DANetHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm(inter_channels),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout(
            0.1, False), nn.Conv(inter_channels, out_channels, 1))

    def execute(self, x):

        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv+sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output


class PAM_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = jt.zeros(1)

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
        proj_query = self.query_conv(x).reshape(
            m_batchsize, -1, width*height).transpose(0, 2, 1)
        proj_key = self.key_conv(x).reshape(m_batchsize, -1, width*height)
        energy = nn.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).reshape(m_batchsize, -1, width*height)

        out = nn.bmm(proj_value, attention.transpose(0, 2, 1))
        out = out.reshape(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = jt.zeros(1)
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.reshape(m_batchsize, C, -1)
        proj_key = x.reshape(m_batchsize, C, -1).transpose(0, 2, 1)
        energy = nn.bmm(proj_query, proj_key)
        #energy_new = jt.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy)
        proj_value = x.reshape(m_batchsize, C, -1)

        out = nn.bmm(attention, proj_value)
        out = out.reshape(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


def main():
    attention_block = DANetHead(64, 64)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
