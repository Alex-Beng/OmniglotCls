import numpy as np
import cv2


def GetHogDsc(image:np.ndarray)->np.ndarray:
    hog = cv2.HOGDescriptor(
        (28, 28), # winsize
        (8, 8), #blockSize,
        (4, 4),# blockStride,
        (4, 4), # cellSize,
        12 # nbins,
        )
    cpt_hog = hog.compute(image).reshape((-1,))
    return cpt_hog
    # print(type(test_hog))
    # print(test_hog.shape)

    
if __name__ == "__main__":
    pass