import numpy as np
from point import Point
from cluster import Cluster

class KMeans:
    def __init__(self, k, dim=None, debug=False, max_iter=200):
        self.k: int = k # cluster count
        self.dim = dim
        self.data = None
        self.clusters = []
        self.debug = debug
        self.maxIter = max_iter

    def initRawData(self, dump):
        self.data = [Point(features) for features in dump]
        if self.dim == None and len(dump) > 0:
            self.dim = len(dump[0])

    def initSampleData(self, dump):
        self.data = dump
        if self.dim == None and len(dump) > 0:
            self.dim = dump[0].dim

    def __recomputeCentroids(self):
        for cluster in self.clusters:
            cluster.calculateCentroid()

    def __reassignMembership(self):
        for cluster in self.clusters:
            cluster.clearMembers()
        for point in self.data:
            distance: dict = {}
            for cluster in self.clusters:
                distance[Point.distance(point, cluster.centroid)] = cluster.id
            trash = list(distance.keys())
            trash.sort()
            junk = distance[trash[0]]
            crap: Cluster = self.clusters[junk]
            crap.addMember(point)

    def train(self):
        assert len(self.data) > 0, "Error: Data is not initialized"
        for i in range(self.k):
            cluster = Cluster(id=i, dim=self.dim)
            cluster.randomizeCentroid()
            self.clusters.append(cluster)
        self.__reassignMembership()
        itr = 0
        while True:
            centroidsPre = [cluster.centroid.features for cluster in self.clusters]
            self.__recomputeCentroids()
            centroidsPost = [cluster.centroid.features for cluster in self.clusters]
            if self.debug:
                print(f"Debug: iteration {itr}")
            self.__reassignMembership()
            itr += 1
            if itr >= self.maxIter or np.allclose(centroidsPre, centroidsPost):
                break
        return self.clusters

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

    kMeans = KMeans(k=4, debug=True)
    kMeans.initRawData(data)
    kMeans.train()
    from silhouette import Silhouette
    sil = Silhouette()
    sil.initKMeans(kMeans)
    print(sil.analyze())
