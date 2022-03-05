import jittor as jt
import jittor.nn as nn


class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c //
                                  self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # attn = nn.bmm(q,k.transpose(0,1,3,2))*self.scale
        attn = nn.bmm_transpose(q, k)*self.scale

        attn = nn.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        out = nn.bmm(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


def main():
    attention_block = MHSA(64)
    input = jt.rand([4, 128, 64])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
