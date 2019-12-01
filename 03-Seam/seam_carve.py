import numpy as np
from skimage import img_as_float
from skimage.color import rgb2gray

def get_energy(img):
    dy = img.copy()
    dy[1:-1, :] = img[2:, :] - img[:-2, :]
    dy[0   , :] = (img[1 , :] - img[ 0 , :])
    dy[-1  , :] = (img[-1, :] - img[-2 , :])
    dx = img.copy()
    dx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    dx[:,   0 ] = (img[:, 1 ] - img[:,  0 ])
    dx[:,  -1 ] = (img[:,-1 ] - img[:, -2 ])
    energy = np.hypot(dx, dy).astype(img.dtype)
    return energy

def get_cumulative_energy(energy):
    cum_energy = energy.copy()
    for i in range(1, cum_energy.shape[0]):
        if cum_energy.shape[1] > 1:
            cum_energy[i, 0] += min(cum_energy[i - 1, 0], cum_energy[i - 1, 1])
            cum_energy[i,-1] += min(cum_energy[i - 1,-1], cum_energy[i - 1,-2])
        else:
            cum_energy[i, 0] += cum_energy[i - 1, 0]
        for j in range(1, cum_energy.shape[1] - 1):
            cum_energy[i, j] += min(cum_energy[i - 1, j - 1], cum_energy[i - 1, j], \
                                cum_energy[i - 1, j + 1])
    return cum_energy

def get_seam_with_min_energy(cum_energy):
    if cum_energy.shape[1] == 1:
        return [i for i in range(cum_energy.shape[0]-1, -1, -1)]
    seam_path = []
    ind_min = np.argmin(cum_energy[-1])
    seam_path.append(ind_min)
    for row in cum_energy[-2::-1]:
        if ind_min == 0:
            ind_min = np.argmin(row[:2])
        elif ind_min == len(row) - 1:
            ind_min = len(row) - 2 + np.argmin(row[-2:])
        else:
            ind_min = ind_min - 1 + np.argmin(row[ind_min - 1: ind_min + 2])
        seam_path.append(ind_min)
    seam_path.reverse()
    return seam_path

def expand_img_by_1px_to_right(img):
    shape = list(img.shape)
    shape[1] += 1
    new_img = np.zeros(tuple(shape)).astype(img.dtype)
    new_img[:, :-1] = img.copy()
    return new_img

def seam_carve(image, mode, mask=None):
    img = image.copy()
    vertical = (mode.split(' ')[0] == 'vertical')
    goal = mode.split(' ')[1]
    if vertical:
        img = img.transpose((1,0,2))
        if mask is not None:
            mask = mask.transpose()
  #  img = img_as_float(img)
    intensity = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    energy = get_energy(intensity)
    if mask is not None:
        energy += img.shape[0] * img.shape[1] * 256 * mask
    cum_energy = get_cumulative_energy(energy)
    seam = get_seam_with_min_energy(cum_energy)
    seam_mask = np.zeros(img.shape[:-1]).astype('int8')
    for i in range(img.shape[0]):
        seam_mask[i, seam[i]] = 1

    _mask = np.array([], dtype='int8')
    if mask is not None:
        _mask = mask.copy()
    if goal == 'shrink':
        for i in range(len(seam)):
            img[i,seam[i]:-1] = img[i, seam[i] + 1:]
            if mask is not None:
                _mask[i, seam[i]:-1] = _mask[i, seam[i] + 1:]
        img = img[:, :-1]
        if mask is not None:
            _mask = _mask[:, :-1]
    elif goal == 'expand':
        img = expand_img_by_1px_to_right(img)
        if mask is not None:
            _mask = expand_img_by_1px_to_right(_mask)
        for i in range(len(seam)):
            s = seam[i]
            right = min(s + 2, img.shape[1] - 1)
            img[i, s + 1:] = img[i, s:-1].copy()
            img[i, s + 1] = (img[i, s] + img[i, right]) / 2.0
            if mask is not None:
                _mask[i, s + 2:] = _mask[i, s + 1:-1]
                _mask[i, s] = 1
                _mask[i, s + 1] = 1

    if vertical:
        img = img.transpose((1,0,2))
        if mask is not None:
            mask = mask.transpose()
        seam_mask = seam_mask.transpose()
    return img, mask, seam_mask
