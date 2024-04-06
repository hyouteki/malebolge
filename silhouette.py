import numpy as np
from point import Point
from kmeans import KMeans

class Silhouette:
    def __init__(self, debug=False):
        self.debug = debug
        self.INT_MAX = float("inf")
        self.__points: list[Point] = []
        self.__clusters: list[Cluster] = []
        self.__aScore: list[float] = []
        self.__bScore: list[float] = []
        self.__sScore: list[float] = []
        self.__distances: list[list[float]] = []

    def __sameCluster(self, i, j):
        point1 = self.__points[i]
        point2 = self.__points[j]
        for cluster in self.__clusters:
            if point1 in cluster.members:
                return point2 in cluster.members
        return False

    def __ai(self):
        self.__aScore = np.zeros(len(self.__points))
        for i in range(len(self.__points)):
            self.__aScore[i] = np.mean([self.__distances[i, j] for j in range(len(self.__points))
                                        if self.__sameCluster(i, j)])

    def __bi(self):
        self.__bScore = np.zeros(len(self.__points))
        minDistance = self.INT_MAX
        for i in range(len(self.__points)):
            point = self.__points[i]
            for j in range(len(self.__clusters)):
                cluster = self.__clusters[j]
                if point not in cluster.members:
                    distance = np.mean([self.__distances[i, k] for k in range(
                        len(self.__points)) if self.__points[k] in cluster.members])
                    if distance < minDistance:
                        minDistance = distance
            self.__bScore[i] = minDistance

    def __si(self):
        self.__sScore = np.zeros(len(self.__points))
        for i in range(len(self.__points)):
            if (self.__bScore[i] == self.INT_MAX):
                self.__sScore[i] = 1
            else:
                self.__sScore[i] = (self.__bScore[i] - self.__aScore[i]) / \
                    max(self.__aScore[i], self.__bScore[i])

    def initKMeans(self, kMeans):
        self.__points = kMeans.data
        self.__clusters = kMeans.clusters
        self.__distances = np.zeros((len(self.__points), len(self.__points)))
        for i in range(len(self.__points)):
            for j in range(len(self.__points)):
                self.__distances[i, j] = Point.distance(self.__points[i],
                                                        self.__points[j])
                self.__distances[j, i] = self.__distances[i, j]

    def analyze(self):
        assert len(self.__points) != 0, "Error: KMeans not initialized"
        self.__ai()
        self.__bi()
        self.__si()
        return sum(self.__sScore)

