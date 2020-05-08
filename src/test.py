import numpy as np
import cv2
from feature import *

def test_show_img():
    a = np.load("../data/Omniglot.npy")
    print(a.shape)
    print(a[0].shape)
    for i in range(10):
        for j in range(10):
            cv2.imshow("ya", a[i][j].astype(np.uint8))
            print(type(a[i][j].astype(np.uint8)))
            cv2.waitKey()

def test_hog_dsc():
    a = np.load("../data/Omniglot.npy")
    for i in range(10):
        for j in range(10):
            ft = GetHogDsc(a[i][j])
            print(ft[:10])


if __name__ == "__main__":
    test_show_img()
    # test_hog_dsc()