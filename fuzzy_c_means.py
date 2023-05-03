import numpy as np
import random


class Point:
    """Data point class"""

    def __init__(self, dim: int):
        """dim :: dimension of the point"""
        self.dim = dim
        self.label = None
        self.junk = None  # can be used to assign any value you like
        self.point: list = [0 for i in range(dim)]

    def setPoint(self, point: list):
        self.point = point

    def distanceTo(self, other) -> float:
        """Euclidean distance between two points"""
        sum = 0
        for i in range(self.dim):
            sum += (self.point[i] - other.point[i])**2
        return sum**(0.5)

    def setLabel(self, label: any):
        self.label = label

    def manhattanDistanceTo(self, other) -> float:
        """Manhattan distance between two points"""
        return sum([abs(self.point[i] - other.point[i]) for i in range(self.dim)])

    @classmethod
    def distance(cls, point1, point2) -> float:
        """Euclidean distance between two points"""
        sum = 0
        for i in range(point1.dim):
            sum += (point1.point[i] - point2.point[i])**2
        return sum**(0.5)

    @classmethod
    def manhattanDistance(cls, point1, point2) -> float:
        """Manhattan distance between two points"""
        return sum([abs(point1.point[i] - point2.point[i]) for i in range(point1.dim)])

    @classmethod
    def toMatrix(cls, object) -> list[list[float]]:
        ret: list[list[float]] = []
        for point in object:
            ret.append(point.point)
        return ret

    @classmethod
    def toPointArray(cls, points, matrix):
        ret: list = []
        dim: int = len(matrix[0])
        for i in range(len(points)):
            points[i].dim = dim
            points[i].setPoint(matrix[i])

    def __str__(self):
        return "{ Point: "+f"{self.point}, Label: {self.label}"+"}"

    def __repr__(self):
        return "{ Point: "+f"{self.point}, Label: {self.label}"+"}"


class Cluster:
    """Cluster class"""

    def __init__(self, id: int, dim: int):
        """
         id :: cluster id
        dim :: dimension of point
        """
        self.id = id
        self.dim = dim
        self.centroid: Point = Point(dim)
        self.members: list[Point] = []

    def randomizeCentroid(self):
        center: list = []
        for i in range(self.dim):
            center.append(random.randint(0, 1))
        self.centroid.setPoint(center)

    def addMember(self, member):
        self.members.append(member)

    def recomputeCentroid(self):
        self.centroid = Point(self.dim)
        for point in self.members:
            for i in range(self.dim):
                self.centroid.point[i] += point.point[i]
        for i in range(self.dim):
            self.centroid.point[i] /= self.dim

    def averagePoint(self) -> Point:
        point: list[float] = [0 for i in range(self.dim)]
        for trash in self.members:
            for i in range(self.dim):
                point[i] += trash.point[i]
        for i in range(self.dim):
            point[i] /= self.dim
        new: Point = Point(self.dim)
        new.setPoint(point)
        return new

    def doomsDay(self):
        self.members: list[Point] = []

    def __str__(self):
        return "{centroid: "+f"{self.centroid.point}"+", members: "+f"{self.members}"+"}"

    def __repr__(self):
        return "{centroid: "+f"{self.centroid.point}"+", members: "+f"{self.members}"+"}"


class FuzzyCMeans:
    def __init__(self, c: int):
        self.c: int = c
        # cluster count
        self.dim = -1
        # dimension / feature count
        self.data: list[Point] = []
        # sample data
        self.clusters: list[Cluster] = []
        # clusters / output
        self.membershipMatrix: list[list[float]] = []
        self.debug = False
        self.MAX_ITR = 200
        self.m = 2
        self.beta = 0.3

    def initRawData(self, dump: list[list[float]]):
        self.data = []
        self.dim = len(dump[0])
        for point in dump:
            trash = Point(self.dim)
            trash.setPoint(point)
            self.data.append(trash)

    def initSampleData(self, dump: list[Point]):
        self.data = dump
        self.dim = len(dump[0].dim)

    def initDimension(self, dump: int):
        self.dim = dump

    def __recomputeCentroids(self):
        for j in range(len(self.clusters)):
            centroid: list[float] = np.zeros(self.dim)
            for i in range(self.dim):
                # print(f"j, i: {j}, {i}")
                centroid[i] = sum([
                    pow(self.membershipMatrix[k][j],
                        self.m) * self.data[k].point[i]
                    for k in range(len(self.data))]) / \
                    sum([pow(self.membershipMatrix[k][j], self.m)
                        for k in range(len(self.data))])
            self.clusters[j].centroid.setPoint(centroid)

    def __reassignMembership(self):
        for i in range(len(self.data)):
            distances: list[float] = [Point.distance(
                self.data[i], self.clusters[j].centroid) for j in range(len(self.clusters))]
            # print(distances)
            for j in range(len(self.clusters)):
                # self.membershipMatrix[i][j] = 1/sum(
                #     [pow(distances[j]/distances[k], 2/self.m - 1) for k in range(len(self.clusters))])
                self.membershipMatrix[i][j] = 1/sum(
                    [distances[j]/distances[k] for k in range(len(self.clusters))])
        # print()

    def doTheJob(self) -> list[Cluster]:
        self.membershipMatrix = np.zeros((len(self.data), self.c))
        for i in range(self.c):
            cluster: Cluster = Cluster(i, self.dim)
            cluster.randomizeCentroid()
            self.clusters.append(cluster)
        while True:
            oldMembershipMatrix = self.membershipMatrix.copy()
            # pprint(self.membershipMatrix)
            self.__reassignMembership()
            self.__recomputeCentroids()
            if np.linalg.norm(self.membershipMatrix - oldMembershipMatrix) < self.beta:
                break


if __name__ == "__main__":
    data: list[list[float]] = [[2.65564218, 1.13781779, 2.],
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

    cmeans = FuzzyCMeans(5)
    cmeans.debug = True
    cmeans.initRawData(data)
    cmeans.doTheJob()
    print(cmeans.membershipMatrix)
