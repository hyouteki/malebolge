import numpy as np
from point import Point
from cluster import Cluster

class FuzzyCMeans:
    def __init__(self, c, debug=False, max_iter=200, m=2, beta=0.3):
        self.c = c # cluster count
        self.dim = None
        self.data = []
        self.clusters = []
        self.membershipMatrix = []
        self.debug = debug
        self.maxIter = max_iter
        self.m = m
        self.beta = beta

    def initRawData(self, dump):
        self.data = [Point(features=features) for features in dump]
        if len(dump) > 0:
            self.dim = len(dump[0])
            
    def initSampleData(self, dump: list[Point]):
        self.data = dump
        if len(dump) > 0:
            self.dim = dump[0].dim

    def __recomputeCentroids(self):
        for j in range(len(self.clusters)):
            centroid = np.zeros(self.dim)
            for i in range(self.dim):
                centroid[i] = sum([pow(self.membershipMatrix[k][j],
                        self.m) * self.data[k].features[i]
                    for k in range(len(self.data))]) / \
                    sum([pow(self.membershipMatrix[k][j], self.m)
                        for k in range(len(self.data))])
            self.clusters[j].centroid.features = centroid

    def __reassignMembership(self):
        for i in range(len(self.data)):
            distances: list[float] = [Point.distance(self.data[i], self.clusters[j].centroid)
                                      for j in range(len(self.clusters))]
            for j in range(len(self.clusters)):
                self.membershipMatrix[i][j] = 1/sum(
                    [distances[j]/distances[k] for k in range(len(self.clusters))])

    def train(self) -> list[Cluster]:
        self.membershipMatrix = np.zeros((len(self.data), self.c))
        for i in range(self.c):
            cluster = Cluster(id=i, dim=self.dim)
            cluster.randomizeCentroid()
            self.clusters.append(cluster)
        while True:
            oldMembershipMatrix = self.membershipMatrix.copy()
            self.__reassignMembership()
            self.__recomputeCentroids()
            if np.linalg.norm(self.membershipMatrix - oldMembershipMatrix) < self.beta:
                break
        return self.membershipMatrix

if __name__ == "__main__":
    data = [[2.65564218, 1.13781779, 2.],
            [3.26703607, 0.45116584, 2.],
            [3.2308542,  0.02495322, 2.],
            [0.58933387, 0.21103187, 0.],
            [1.36292299, 3.25242328, 1.],
            [0.23334512, 0.79804703, 0.],
            [4.26280164, 2.41056936, 3.],
            [3.58650571, 0.92836418, 2.],
            [1.56042522, 2.30161603, 1.],
            [1.0651578,  1.36777427, 0.],
            [3.68544639, 2.86744397, 3.],
            [1.76788308, 3.29652383, 1.],
            [1.19406845, 1.28510104, 0.],
            [0.20820074, 1.16361149, 0.],
            [2.13059544, 3.30791237, 3.],
            [2.95923093, 2.6636872, 3.],
            [2.7144565, 2.49166562, 3.],
            [1.28480937, 1.09334075, 0.],
            [3.61619512, 1.71713123, 2.],
            [0.65144478, 3.47713109, 1.],
            [3.07201845, 3.52020652, 3.],
            [0.43963793, 3.09845707, 1.],
            [0.9593578, 3.07223414, 1.],
            [3.00656093, 1.89689878, 2.]]    
    cmeans = FuzzyCMeans(c=5, debug=True)
    cmeans.initRawData(data)
    print(cmeans.train())
