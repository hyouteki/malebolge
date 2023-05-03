import numpy as np


class Point:
    def __init__(self, id: int = -1,
                 features: list[float] = list(), label: int = -1):
        self.id = id
        self.features = features
        self.label = label

    def __str__(self):
        return f"[Id :: {self.id}, Features :: {self.features}, Label :: {self.label}]"

    def __repr__(self):
        return f"[Id :: {self.id}, Features :: {self.features}, Label :: {self.label}]"

    @classmethod
    def distance(cls, point1, point2):
        """Euclidean distance between two points"""
        sum = 0
        for i in range(point1.dim):
            sum += (point1.features[i] - point2.features[i])**2
        return sum**(0.5)

    @classmethod
    def subtractPointToNumpyArray(cls, point1, point2):
        return np.subtract(np.array(point1.features), np.array(point2.features))

    @classmethod
    def addPoint(cls, point1, point2):
        return Point(features=[point1.features[i] + point2.features[i]
                               for i in range(len(point1.features))])


class Cluster:
    def __init__(self, members: list[Point] = list()):
        self.id = members[0].label
        self.members = members

    def calculateCentroid(self) -> Point:
        trash = np.array([float(0)
                         for _ in range(len(self.members[0].features))])
        for member in self.members:
            trash += np.array(member.features)
        trash = trash/len(self.members)
        return Point(features=trash.tolist(), label=self.id)

    def addMember(self, point) -> bool:
        if point.label == self.id:
            self.members.append(point)
            return True
        else:
            return False

    def __str__(self):
        return f"[Id :: {self.id}, member size :: {len(self.members)}]"

    def __repr__(self):
        return f"[Id :: {self.id}, member size :: {len(self.members)}]"

    def calculateSj(self):
        centroid = self.calculateCentroid()
        sj = np.matrix([[float(0) for _ in range(len(self.members[0].features))]
                       for _ in range(len(self.members[0].features))])
        for member in self.members:
            sub = np.matrix(Point.subtractPointToNumpyArray(member, centroid))
            trans = sub.T
            sj += np.dot(trans, sub)
        return sj


class LDA:
    def __init__(self, k: int = -1, dataset: list[Point] = list()):
        self.k = k
        self.__dataset = dataset
        self.__clusters: list[Cluster] = list()
        self.debug = False
        self.transformationMatrix = list()

    def __calculateM(self) -> Point:
        trash = np.array([float(0)
                         for _ in range(len(self.__dataset[0].features))])
        for point in self.__dataset:
            trash += np.array(point.features)
        trash = trash/len(self.__dataset)
        return Point(features=trash.tolist())

    def __makeClusters(self) -> None:
        clusterMap: dict[int, Cluster] = dict()
        for point in self.__dataset:
            if point.label not in clusterMap.keys():
                cluster: Cluster = Cluster(members=[point])
                clusterMap[point.label] = cluster
            else:
                clusterMap[point.label].addMember(point)
        self.__clusters = [cluster for cluster in clusterMap.values()]

    def __calculateMjs(self) -> list[Point]:
        mjs: list[Point] = list()
        for cluster in self.__clusters:
            mjs.append(cluster.calculateCentroid())
        return mjs

    def __calculateSw(self):
        sw = np.matrix([[float(0) for _ in range(len(self.__dataset[0].features))]
                       for _ in range(len(self.__dataset[0].features))])
        for cluster in self.__clusters:
            sw += np.matrix(cluster.calculateSj())
        return sw

    def __calculateSb(self):
        sb = np.matrix([[float(0) for _ in range(len(self.__dataset[0].features))]
                       for _ in range(len(self.__dataset[0].features))])
        m: Point = self.__calculateM()
        mjs: list[Point] = self.__calculateMjs()
        for i in range(len(self.__clusters)):
            mj: Point = mjs[i]
            nj: int = len(self.__clusters[i].members)
            sub = np.matrix(Point.subtractPointToNumpyArray(mj, m))
            trans = sub.T
            sb += np.dot(trans, sub)*nj
        return sb

    def __transformToFeatureMatrix(self):
        return np.matrix([point.features for point in self.__dataset])

    def transform(self, dump) -> list:
        return np.dot(dump, self.transformationMatrix).tolist()

    def doTheJob(self) -> list:
        self.__makeClusters()
        if self.debug:
            print(self.__clusters)
        sw = self.__calculateSw()
        if self.debug:
            print(sw)
        # print(sw)
        sb = self.__calculateSb()
        # print(sb)
        swInv = np.linalg.inv(sw)
        a = np.dot(swInv, sb)
        eigenvalues, eigenvectors = np.linalg.eig(a)
        if self.debug:
            print(eigenvalues)
        if self.debug:
            print(eigenvectors)
        largest_indices = np.argsort(eigenvalues)[::-1][:self.k]
        if self.debug:
            print(largest_indices)
        if self.debug:
            print(eigenvalues)
        if self.debug:
            print(eigenvectors)
        largestEigenvalues = eigenvalues[largest_indices]
        largestEigenvectors = eigenvectors[:, largest_indices]
        if self.debug:
            print(largestEigenvalues)
        if self.debug:
            print(largestEigenvectors)
        self.transformationMatrix = largestEigenvectors
        return self.transform(self.__transformToFeatureMatrix())
