#from scipy import weave
import numpy as np
import cv2
import sys
import imutils
from numba import jit

@jit
def _thinningIteration(im, iter_):
    M = np.zeros(im.shape, np.uint8)
    h, w = im.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            p2 = im[i - 1, j]
            p3 = im[i - 1, j + 1]
            p4 = im[i, j + 1]
            p5 = im[i + 1, j + 1]
            p6 = im[i + 1, j]
            p7 = im[i + 1, j - 1]
            p8 = im[i, j - 1]
            p9 = im[i - 1, j - 1]
            A = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            m1 = (p2 * p4 * p6) if (iter_ == 0) else (p2 * p4 * p8)
            m2 = (p4 * p6 * p8) if (iter_ == 0) else (p2 * p6 * p8)
            if A == 1 and B >= 2 and B <=6 and m1 == 0 and m2 == 0:
                M[i, j] = 1

    return im & ~M

def thinning(src):
	dst = src.copy() / 255
	prev = np.zeros(src.shape[:2], np.uint8)
	diff = None

	while True:
		dst = _thinningIteration(dst, 0)
		dst = _thinningIteration(dst, 1)
		diff = np.absolute(dst - prev)
		prev = dst.copy()
		if np.sum(diff) == 0:
			break

	return dst * 255

def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])

    filtered = cv2.filter2D(skel,-1,kernel)

    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 255
    return out

if __name__ == "__main__":

	src = cv2.imread("circuit6.jpg")
	src = imutils.resize(src,width=640)
	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

	img = cv2.GaussianBlur(gray,(9,9),0)
	th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
			cv2.THRESH_BINARY_INV,11,2)
	bw = thinning(th)

	ends = skeleton_endpoints(bw)
	cv2.imshow("thinning", bw)
	cv2.imshow("ends", ends)
	cv2.waitKey()
