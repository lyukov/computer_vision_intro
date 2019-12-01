from itertools import product
import numpy as np


def find_shift(img, img1):
    first_pos = (0,0)
    max_shift = 15
    height = img.shape[0]
    width  = img.shape[1]
    if height > 200:
        first_pos = find_shift(img[::2, ::2], img1[::2, ::2])
        max_shift = 1
    first_pos = (first_pos[0] * 2, first_pos[1] * 2)
    shift = (0, 0)
    min_mse = 1e9
    for i in range(first_pos[0] - max_shift, first_pos[0] + max_shift + 1):
        for j in range(first_pos[1] - max_shift, first_pos[1] + max_shift + 1):
            l1, l2, u1, u2 = 0, 0, 0, 0
            r1, r2, d1, d2 = width, width, height, height
            if i < 0:
                u1, d2 = -i, i
            elif i > 0:
                u2, d1 = i, -i
            if j < 0:
                l1, r2 = -j, j
            elif j > 0:
                l2, r1 = j, -j
            c = height // 3
            first  = img [(u1+c):(d1-c), (l1+c):(r1-c)]
            second = img1[(u2+c):(d2-c), (l2+c):(r2-c)]
            mse = ((first - second) ** 2).mean()
            if mse < min_mse:
                min_mse = mse
                shift = (i, j)
    return shift


def align(img, g_coord):
    row_g, col_g = g_coord
    rows = img.shape[0]//3
    rgb = []
    rgb.append(img[rows*2:rows*3, :].copy())
    rgb.append(img[rows  :rows*2, :].copy())
    rgb.append(img[0     :rows  , :].copy())
    crop0 = int(rgb[0].shape[0]*0.05)
    crop1 = int(rgb[0].shape[1]*0.05)
    for chnl in range(len(rgb)):
        rgb[chnl] = rgb[chnl][crop0:-crop0, crop1:-crop1]
    shifts = [(0,0), (0,0), (0,0)]
    for chnl in 0,2:
        shift = find_shift(rgb[chnl], rgb[1])
        shifts[chnl] = shift
    combined = np.zeros_like(rgb[0]).astype('uint8')
    b_coord = (row_g - rows - shifts[2][0], col_g - shifts[2][1])
    r_coord = (row_g + rows - shifts[0][0], col_g - shifts[0][1])
    return combined, b_coord, r_coord
