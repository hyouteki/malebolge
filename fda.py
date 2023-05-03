import numpy as np


class Point:
    def __init__(self, id: int = -1,
                 features: list = list(), label: int = -1):
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

    def calculateSnot(self):
        centroid = self.calculateCentroid()
        sj = np.matrix([[float(0) for _ in range(len(self.members[0].features))]
                       for _ in range(len(self.members[0].features))])
        for member in self.members:
            sub = np.matrix(Point.subtractPointToNumpyArray(member, centroid))
            trans = sub.T
            sj += np.dot(trans, sub)
        sj = sj/(len(self.members)-1)
        return sj

    def __str__(self):
        return f"[Id :: {self.id}, member size :: {len(self.members)}]"

    def __repr__(self):
        return f"[Id :: {self.id}, member size :: {len(self.members)}]"


class FDA:
    def __init__(self, dataset: list[Point] = list(), debug: bool = False):
        self.dataset = dataset
        self.debug = debug
        self.classes = list()
        self.transformationMatrix = list()

    def __initClasses(self):
        classMap: dict[int, Cluster] = dict()
        for point in self.dataset:
            if point.label in classMap.keys():
                classMap[point.label].addMember(point)
            else:
                classMap[point.label] = Cluster([point])
        self.classes = [trash for trash in classMap.values()]

    def __initTransformationMatrix(self):
        m0 = np.array(self.classes[0].calculateCentroid().features)
        m1 = np.array(self.classes[1].calculateCentroid().features)
        n0 = self.classes[0].members.__len__()
        n1 = self.classes[1].members.__len__()
        s0 = self.classes[0].calculateSnot()
        s1 = self.classes[1].calculateSnot()
        self.transformationMatrix = np.dot(
            np.linalg.inv(np.add(n0*s0, n1*s1)),
            np.subtract(m0, m1)
        )

    def transformData(self, dump: list[Point]):
        return [np.dot(self.transformationMatrix, np.matrix(point.features).T).tolist()[0][0]
                for point in dump]

    def doTheJob(self):
        self.__initClasses()
        # pprint(self.classes)
        self.__initTransformationMatrix()
        # print(self.transformationMatrix)
        return self.transformData(self.dataset)
