import cv2
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

x_file = open("../data/x.bin", "rb")
y_file = open("../data/y.bin", "rb")

x = pickle.load(x_file)
y = pickle.load(y_file)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.4, random_state=1)

# classifier = KMeans(n_clusters=100)
# classifier = AdaBoostClassifier()
# 0.01
# classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100)) 
# 0.71
# classifier = GradientBoostingClassifier()
# 0.14
# classifier = KNeighborsClassifier(n_neighbors=5)
# 0.65
# classifier = RandomForestClassifier()
# 0.35
# classifier = DecisionTreeClassifier()
# 0.22
# classifier = SVC(kernel='linear', decision_function_shape='ovr')
# 0.78
# classifier = LinearSVC()
# 0.74
# classifier = SVC(kernel='linear', decision_function_shape='ovo')
# 0.78
classifier = SVC(kernel='linear', decision_function_shape='ovr')



classifier.fit(x_train, y_train)

print(classifier.score(x_train, y_train))
print(classifier.score(x_test, y_test))
