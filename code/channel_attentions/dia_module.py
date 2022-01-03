# DIANet: Dense-and-Implicit Attention Network (AAAI 2020)
import jittor as jt
from jittor import nn


class small_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """"Constructor of the class"""
        super(small_cell, self).__init__()
        self.seq = nn.Sequential(nn.Linear(input_size, input_size // 4),
                                 nn.ReLU(),
                                 nn.Linear(input_size // 4, 4 * hidden_size))

    def execute(self, x):
        return self.seq(x)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout=0.1):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            if i == 0:
                # ih.append(nn.Linear(input_size, 4 * hidden_size))
                ih.append(small_cell(input_size, hidden_size))
                # hh.append(nn.Linear(hidden_size, 4 * hidden_size))
                hh.append(small_cell(hidden_size, hidden_size))
            else:
                ih.append(nn.Linear(hidden_size, 4 * hidden_size))
                hh.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def execute(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
            i_gate = i_gate.sigmoid()
            f_gate = f_gate.sigmoid()
            c_gate = jt.tanh(c_gate)
            o_gate = o_gate.sigmoid()
            ncx = (f_gate * cx) + (i_gate * c_gate)
            # nhx = o_gate * torch.tanh(ncx)
            nhx = o_gate * ncx.sigmoid()
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = jt.stack(hy, 0), jt.stack(
            cy, 0)  # number of layer * batch * hidden
        return hy, cy


class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()
        self.lstm = LSTMCell(channel, channel, 1)

        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()

    def execute(self, x):
        org = x
        seq = self.GlobalAvg(x)
        seq = seq.view(seq.size(0), seq.size(1))
        ht = jt.zeros((1, seq.size(0), seq.size(
            1)))  # 1 mean number of layers
        ct = jt.zeros((1, seq.size(0), seq.size(1)))
        ht, ct = self.lstm(seq, (ht, ct))  # 1 * batch size * length
        # ht = self.sigmoid(ht)
        x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))
        x += org
        x = self.relu(x)

        return x  # , list


def main():
    attention_block = Attention(64)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
