from skimage.io import imread, imsave, imshow
from skimage.filters import sobel_h, sobel_v
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn import svm
import numpy as np


def grad_magn_orient(img):
    dx = sobel_h(img)
    dy = sobel_v(img)
    return np.hypot(dx, dy), np.pi + np.arctan2(dx, dy)

def hog(img):
    bin_count = 8
    N_SEGMENTS = 14
    seg_h = (img.shape[0] + N_SEGMENTS - 1) // N_SEGMENTS
    seg_w = (img.shape[1] + N_SEGMENTS - 1) // N_SEGMENTS
    indent = 3
    hist = np.zeros((N_SEGMENTS-2*indent, N_SEGMENTS-2*indent, bin_count))
    magn, orient = grad_magn_orient(img)
    for i in range(indent, N_SEGMENTS - indent):
        for j in range(indent, N_SEGMENTS - indent):
            orient_seg = orient[i*seg_h : (i+1)*seg_h, j*seg_w : (j+1)*seg_w]
            magn_seg = magn[i*seg_h : (i+1)*seg_h, j*seg_w : (j+1)*seg_w]
            result = np.histogram(
                orient_seg,
                bins=bin_count,
                range=(-np.pi, np.pi),
                weights=magn_seg
            )[0]
            result /= (np.linalg.norm(result) + 1e-6)
            hist[i - indent, j - indent] = result
    return hist.flatten()

def extract_hog(img):
    image_resized = resize(img, (140, 140), anti_aliasing=True)
    return hog(rgb2gray(image_resized))

def fit_and_classify(train_features, train_labels, test_features):
    classifier = svm.SVC(gamma="scale", kernel='rbf', C=16.0)
    classifier.fit(train_features, train_labels)
    return classifier.predict(test_features)
