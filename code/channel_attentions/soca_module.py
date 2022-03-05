# Second-Order Attention Network for Single Image Super-Resolution (CVPR 2019)
import jittor as jt
from jittor import nn, Function


class Covpool(Function):
    def execute(self, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h*w
        x = x.reshape(batchSize, dim, M)
        I_hat = (-1./M/M)*jt.ones((M, M)) + (1./M)*jt.init.eye((M, M))
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1)
        y = nn.bmm(nn.bmm(x, I_hat), x.transpose(0, 2, 1))
        self.save_vars = (input, I_hat)
        return y

    def grad(self, grad_output):
        input, I_hat = self.save_vars
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h*w
        x = x.reshape(batchSize, dim, M)
        grad_input = grad_output + grad_output.transpose(0, 2, 1)
        grad_input = nn.bmm(nn.bmm(grad_input, x), I_hat)
        grad_input = grad_input.reshape(batchSize, dim, h, w)
        return grad_input


class Sqrtm(Function):
    def execute(self, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        I3 = 3.0*jt.init.eye((dim, dim)).view(1,
                                              dim, dim).repeat(batchSize, 1, 1)
        normA = (1.0/3.0)*x.matmul(I3).sum(dim=1).sum(dim=1)
        A = x.divide(normA.view(batchSize, 1, 1).expand_as(x))
        Y = jt.zeros((batchSize, iterN, dim, dim))
        Y.requires_grad = False
        Z = jt.init.eye((dim, dim)).view(
            1, dim, dim).repeat(batchSize, iterN, 1, 1)
        if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:, 0, :, :] = nn.bmm(A, ZY)
        else:
            ZY = 0.5*(I3 - A)
            Y[:, 0, :, :] = nn.bmm(A, ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN-1):
                ZY = 0.5*nn.bmm(I3 - Z[:, i-1, :, :], Y[:, i-1, :, :])
                Y[:, i, :, :] = nn.bmm(Y[:, i-1, :, :], ZY)
                Z[:, i, :, :] = nn.bmm(ZY, Z[:, i-1, :, :])
            ZY = nn.bmm(
                nn.bmm(0.5*Y[:, iterN-2, :, :], I3 - Z[:, iterN-2, :, :]), Y[:, iterN-2, :, :])
        y = ZY*jt.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        self.save_vars = (input, A, ZY, normA, Y, Z)
        self.iterN = iterN
        return y

    def grad(self, grad_output):
        input, A, ZY, normA, Y, Z = self.save_vars
        iterN = self.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        der_postCom = grad_output * \
            jt.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (
            grad_output*ZY).sum(dim=1).sum(dim=1).divide(2*jt.sqrt(normA))
        I3 = 3.0*jt.init.eye((dim, dim)).view(1, dim,
                                              dim).repeat(batchSize, 1, 1)
        if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
        else:
            dldY = 0.5*(nn.bmm(der_postCom, I3 - nn.bmm(Y[:, iterN-2, :, :], Z[:, iterN-2, :, :])) - nn.bmm(
                nn.bmm(Z[:, iterN-2, :, :], Y[:, iterN-2, :, :]), der_postCom))
            dldZ = -0.5*nn.bmm(
                nn.bmm(Y[:, iterN-2, :, :], der_postCom), Y[:, iterN-2, :, :])
            for i in range(iterN-3, -1, -1):
                YZ = I3 - nn.bmm(Y[:, i, :, :], Z[:, i, :, :])
                ZY = nn.bmm(Z[:, i, :, :], Y[:, i, :, :])
                dldY_ = 0.5*(nn.bmm(dldY, YZ) -
                             nn.bmm(nn.bmm(Z[:, i, :, :], dldZ), Z[:, i, :, :]) -
                             nn.bmm(ZY, dldY))
                dldZ_ = 0.5*(nn.bmm(YZ, dldZ) -
                             nn.bmm(nn.bmm(Y[:, i, :, :], dldY), Y[:, i, :, :]) -
                             nn.bmm(dldZ, ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5*(nn.bmm(dldY, I3 - A) - dldZ - nn.bmm(A, dldY))
        grad_input = der_NSiter.divide(
            normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.matmul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i]
                                    - grad_aux[i] / (normA[i] * normA[i])) * jt.ones((dim,)).diag()
        return grad_input, None


class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.CovpoolLayer = Covpool()
        self.SqrtmLayer = Sqrtm()

    def execute(self, x):
        b, c, h, w = x.shape

        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]

        # MPN-COV
        cov_mat = self.CovpoolLayer(x_sub)
        cov_mat_sqrt = self.SqrtmLayer(cov_mat, 5)

        cov_mat_sum = jt.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(b, c, 1, 1)

        y_cov = self.conv_du(cov_mat_sum)

        return y_cov*x


def main():
    attention_block = SOCA(64)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    jt.grad(output, input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
