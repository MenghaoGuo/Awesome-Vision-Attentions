# Attention augmented convolutional networks (ICCV 2019)
import jittor as jt
from jittor import nn


class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, relative):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative

        self.conv_out = nn.Conv2d(
            self.in_channels, self.out_channels - self.dv, self.kernel_size, padding=1)

        self.qkv_conv = nn.Conv2d(
            self.in_channels, 2 * self.dk + self.dv, kernel_size=1)

        self.attn_out = nn.Conv2d(self.dv, self.dv, 1)

    def execute(self, x):
        # Input x
        # (batch_size, channels, height, width)
        batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(
            x, self.dk, self.dv, self.Nh)
        logits = jt.matmul(flat_q.transpose(2, 3), flat_k)

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = nn.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = jt.matmul(weights, flat_v.transpose(2, 3))
        attn_out = jt.reshape(
            attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return jt.concat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        N, _, H, W = x.size()
        qkv = self.qkv_conv(x)
        q, k, v = jt.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = jt.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = jt.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = jt.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = jt.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return jt.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = jt.transpose(q, 2, 4).transpose(2, 3)

        key_rel_w = jt.randn((2 * W - 1, dk))
        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, "w")

        key_rel_h = jt.randn((2 * H - 1, dk))
        rel_logits_h = self.relative_logits_1d(
            jt.transpose(q, 2, 3), key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = jt.matmul(q, rel_k.transpose(0, 1))
        rel_logits = jt.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = jt.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = jt.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = jt.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = jt.transpose(
                rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = jt.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = jt.zeros((B, Nh, L, 1))
        x = jt.concat((x, col_pad), dim=3)

        flat_x = jt.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = jt.zeros((B, Nh, L - 1))
        flat_x_padded = jt.concat((flat_x, flat_pad), dim=2)

        final_x = jt.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


def main():
    attention_block = AugmentedConv(64, 64, 3, 40, 4, 4, True)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
