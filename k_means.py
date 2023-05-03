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


class KMeans:
    def __init__(self, k: int):
        self.k: int = k
        # cluster count

        self.dim = -1
        # dimension / feature count

        self.data: list[Point] = []
        # sample data

        self.clusters: list[Cluster] = []
        # clusters / output

        self.debug = False

        self.MAX_ITR = 200

    def help(self):
        print("""
Initialize dimension using                   :: initDimension(dim: int)
Intialize Sample data in raw form by using   :: initRawData(data: list[list[float]])
Intialize Sample data in point form by using :: initSampleData(data: list[Point])
Run the algorithm using                      :: doTheJob()
        """)

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
        for cluster in self.clusters:
            cluster.recomputeCentroid()

    def __reassignMembership(self):
        for cluster in self.clusters:
            cluster.doomsDay()
        for point in self.data:
            distance: dict = {}
            for cluster in self.clusters:
                distance[Point.distance(
                    point, cluster.centroid)] = cluster.id
            trash = list(distance.keys())
            trash.sort()
            junk = distance[trash[0]]
            crap: Cluster = self.clusters[junk]
            crap.addMember(point)

    def doTheJob(self) -> list[Cluster]:
        if (len(self.data) == 0):
            print("Sample data is not initialized")
            exit(1)
        if (self.dim == -1):
            print("Data point dimension is not initialized")
            exit(1)
        # initializing the clusters
        for i in range(self.k):
            cluster: Cluster = Cluster(i, self.dim)
            cluster.randomizeCentroid()
            self.clusters.append(cluster)
        itr: int = 1
        if (self.debug):
            print(f"At iteration {itr}")
            print(self.clusters)
        self.__reassignMembership()
        itr: int = 0
        while (True):  # until convergence
            # before shifted centroids
            centroidsPre = [
                cluster.centroid.point for cluster in self.clusters]
            self.__recomputeCentroids()
            centroidsPost = [
                cluster.centroid.point for cluster in self.clusters]
            itr += 1
            if (self.debug):
                print(f"At iteration {itr}")
                print(self.clusters, end="\n\n")
            self.__reassignMembership()
            # convergence check
            i += 1
            if (i == self.MAX_ITR):
                break
            if (np.allclose(centroidsPre, centroidsPost)):
                break
        # self.clusters.sort(key=lambda x: x.members, reverse=True)
        # index: int = 0
        # for cluster in self.clusters:
        #     if (len(cluster.members) == 0):
        #         break
        #     index += 1
        # self.clusters = self.clusters[:index]
        return self.clusters


class Silhouette:
    def __init__(self):
        self.debug = False
        self.INT_MAX = float("inf")
        self.__points: list[Point] = []
        self.__clusters: list[Cluster] = []
        self.__aScore: list[float] = []
        self.__bScore: list[float] = []
        self.__sScore: list[float] = []
        self.__distances: list[list[float]] = []

    def __sameCluster(self, i: int, j: int) -> bool:
        point1: Point = self.__points[i]
        point2: Point = self.__points[j]
        for cluster in self.__clusters:
            if point1 in cluster.members:
                return point2 in cluster.members
        return False

    def __ai(self):
        self.__aScore = np.zeros(len(self.__points))
        for i in range(len(self.__points)):
            self.__aScore[i] = np.mean([self.__distances[i, j] for j in range(
                len(self.__points)) if self.__sameCluster(i, j)])

    def __bi(self):
        self.__bScore = np.zeros(len(self.__points))
        minDistance: float = self.INT_MAX
        for i in range(len(self.__points)):
            point: Point = self.__points[i]
            for j in range(len(self.__clusters)):
                cluster: Cluster = self.__clusters[j]
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

    def initKMeans(self, kMeans: KMeans):
        self.__points = kMeans.data
        self.__clusters = kMeans.clusters
        self.__distances = np.zeros(
            (len(self.__points), len(self.__points)))
        for i in range(len(self.__points)):
            for j in range(len(self.__points)):
                self.__distances[i, j] = Point.distance(
                    self.__points[i], self.__points[j])
                self.__distances[j, i] = self.__distances[i, j]

    def doTheJob(self) -> float:
        """returns silhouette score"""
        if (len(self.__points) == 0):
            print("K means is not initialized. Initialize it by initKMeans()")
            exit(1)
        self.__ai()
        self.__bi()
        self.__si()
        print()
        return sum(self.__sScore)


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

    kMeans = KMeans(4)
    kMeans.debug = True
    kMeans.initRawData(data)
    kMeans.doTheJob()
    # kMeans.help()
    sil = Silhouette()
    sil.initKMeans(kMeans)
    print(sil.doTheJob())
