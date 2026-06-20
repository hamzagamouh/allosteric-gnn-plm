import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def make_knn(X, y, n_neighbors=30, metric="euclidean"):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X, y.ravel())
    return knn
