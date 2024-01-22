import numpy as np
import cv2
import sys
from numba import jit

@jit(nopython=True)
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


if __name__ == "__main__":
	src = cv2.imread("test.png")
	bw = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
	cv2.imshow("src", bw)
	bw2 = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY)
	bw2 = thinning(bw2)
	cv2.imshow("thinning", bw2)
	cv2.waitKey()
