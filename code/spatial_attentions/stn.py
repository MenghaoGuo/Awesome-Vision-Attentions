import jittor as jt
from jittor import nn
import numpy as np


def get_pixel_value(img, x, y):
    B, C, H, W = img.shape
    return img.reindex([B, C, H, W], ['i0', 'i1', '@e0(i0, i2, i3)','@e1(i0, i2, i3)'], extras=[x, y])


def affine_grid_generator(height, width, theta):
    num_batch = theta.shape[0]
    
    # create normalized 2D grid
    x = jt.linspace(-1.0, 1.0, width)
    y = jt.linspace(-1.0, 1.0, height)
    x_t, y_t = jt.meshgrid(x, y)

    # flatten
    x_t_flat = x_t.reshape(-1)
    y_t_flat = y_t.reshape(-1)
    print(x_t.shape)
    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = jt.ones_like(x_t_flat)
    sampling_grid = jt.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = sampling_grid.unsqueeze(0).expand(num_batch, -1, -1)


    # transform the sampling grid - batch multiply
    batch_grids = jt.matmul(theta, sampling_grid)

    # reshape to (num_batch, H, W, 2)
    batch_grids = batch_grids.reshape(num_batch, 2, height, width)
    return batch_grids


def bilinear_sampler(img, x, y):
    B, C, H ,W = img.shape
    max_y = H - 1
    max_x = W - 1

    # rescale x and y to [0, W-1/H-1]
    x = 0.5 * (x + 1.0) * (max_x-1)
    y = 0.5 * (y + 1.0) * (max_y-1)

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = jt.floor(x).astype('int32')
    x1 = x0 + 1
    y0 = jt.floor(y).astype('int32')
    y1 = y0 + 1

    x0 = jt.minimum(jt.maximum(0, x0), max_x)
    x1 = jt.minimum(jt.maximum(0, x1), max_x)
    y0 = jt.minimum(jt.maximum(0, y0), max_y)
    y1 = jt.minimum(jt.maximum(0, y1), max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # compute output
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    return out

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
    
    def execute(self, x1, theta):
        B, C, H, W = x1.shape
        theta = theta.reshape(-1, 2, 3)
        
        batch_grids = affine_grid_generator(H, W, theta)
        
        x_s = batch_grids[:, 0, :, :]
        y_s = batch_grids[:, 1, :, :]

        out_fmap = bilinear_sampler(x1, x_s, y_s)

        return out_fmap


def main():
    stn = STN()
    x = jt.randn(1, 3, 224, 224)
    theta = jt.array(np.random.uniform(0,1,(1,6)))
    y = stn(x, theta)
    print(y)

if __name__ == "__main__":
    main()
