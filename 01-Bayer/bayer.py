import numpy as np
from scipy import signal
from skimage import img_as_float


def get_bayer_masks(n_rows, n_cols): 
    r_mask = [[0, 1], [0, 0]]
    g_mask = [[1, 0], [0, 1]]
    b_mask = [[0, 0], [1, 0]]
    return np.array(list(
                map(
                    lambda mask:
                        np.tile(np.array(mask), (n_rows // 2 + 1, n_cols // 2 + 1))[:n_rows, :n_cols],
                    [r_mask, g_mask, b_mask]
                )
           )).transpose((1,2,0))


def get_colored_img(raw_img):
    masks = get_bayer_masks(*raw_img.shape)
    return (masks.transpose((2,0,1)) * raw_img).transpose((1,2,0))


def conv2d_with_ones(img):
    image = img.transpose((2,0,1))
    ker = np.ones((3,3)).astype('ubyte')
    return np.array(list(
                map(
                    lambda x:
                        signal.convolve(x, ker)[1:-1, 1:-1],
                    image
                )
           )).transpose((1,2,0))


def bilinear_interpolation(colored_img):
    #f_colored_img = img_as_float(colored_img.astype('ubyte'))
    masks = get_bayer_masks(*colored_img.shape[:-1])
    result = conv2d_with_ones(colored_img) / conv2d_with_ones(masks) * (1 - masks) + colored_img
    return result.astype('ubyte')


def compute_psnr(img_pred, img_gt):
    mse = ((img_pred - img_gt) ** 2).mean()
    if mse == 0:
        raise ValueError("")
    MAX_PIX = img_gt.max()
    return 10.0 * np.log10(MAX_PIX ** 2 / mse)
