import cv2
import numpy as np
import pickle

from sklearn.decomposition import PCA


from feature import *

x = []
y = []

data_set = np.load("../data/Omniglot.npy")

class_num = 1623
num_per_class = 20

for i in range(class_num):
    for j in range(num_per_class):
        y.append(i)
        x.append(GetHogDsc(data_set[i][j]))

pca = PCA(n_components = 300)
pca.fit(x)
x = pca.transform(x)

x_file = open("../data/x.bin", "wb")
y_file = open("../data/y.bin", "wb")
pickle.dump(x, x_file)
pickle.dump(y, y_file)