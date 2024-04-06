import numpy as np
from point import Point
from cluster import Cluster

class LDA:
    def __init__(self, k, data, debug=False):
        self.k = k
        self.data = data
        self.clusters = []
        self.debug = debug
        self.transformationMatrix = []

    def __calculateM(self):
        trash = np.array([float(0) for _ in range(self.data[0].dim)])
        for point in self.data:
            trash += np.array(point.features)
        trash = trash/len(self.data)
        return Point(features=trash.tolist())

    def __makeClusters(self):
        clusterMap = dict()
        for point in self.data:
            if point.label not in clusterMap.keys():
                cluster: Cluster = Cluster(id=point.label, members=[point])
                clusterMap[point.label] = cluster
            else:
                clusterMap[point.label].addMember(point)
        self.clusters = list(clusterMap.values())

    def __calculateMjs(self) -> list[Point]:
        return [cluster.calculateCentroid() for cluster in self.clusters]

    def __calculateSw(self):
        assert len(self.data) > 0
        sw = np.matrix([[float(0) for _ in range(self.data[0].dim)]
                        for _ in range(self.data[0].dim)])
        for cluster in self.clusters:
            sw += np.matrix(cluster.calculateSj())
        return sw

    def __calculateSb(self):
        assert len(self.data) > 0
        sb = np.matrix([[float(0) for _ in range(self.data[0].dim)]
                        for _ in range(self.data[0].dim)])
        m = self.__calculateM()
        mjs = self.__calculateMjs()
        for i, cluster in enumerate(self.clusters):
            mj: Point = mjs[i]
            nj: int = len(cluster.members)
            sub = np.matrix(Point.subtractPointToNumpyArray(mj, m))
            trans = sub.T
            sb += np.dot(trans, sub)*nj
        return sb

    def __transformToFeatureMatrix(self):
        return np.matrix(Point.toMatrix(self.data))

    def transform(self, dump):
        return np.dot(dump, self.transformationMatrix).tolist()

    def analyze(self) -> list:
        self.__makeClusters()
        if self.debug:
            print(f"Debug: clusters\n{self.clusters}\n")
        sw = self.__calculateSw()
        if self.debug:
            print(f"Debug: sw\n{sw}\n")
        sb = self.__calculateSb()
        swInv = np.linalg.inv(sw)
        a = np.dot(swInv, sb)
        eigenvalues, eigenvectors = np.linalg.eig(a)
        largestIndices = np.argsort(eigenvalues)[::-1][:self.k]
        if self.debug:
            print(f"Debug: largest indices\n{largestIndices}\n")
        largestEigenvectors = eigenvectors[:, largestIndices]
        if self.debug:
            print(f"Debug: largest eigenvectors\n{largestEigenvectors}\n")
        self.transformationMatrix = largestEigenvectors
        return self.transform(self.__transformToFeatureMatrix())
