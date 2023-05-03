import numpy as np
from pprint import pprint


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


class LogisticRegression:
    def __init__(self, dataset: list[Point] = list(), debug: bool = False,
                 learningRate: float = 0.02):
        self.debug = debug
        self.dataset = dataset
        self.learningRate = learningRate
        self.maxIterationLimit = 1000
        self.weights: list[float] = [float(0) for _ in
                                     range(len(self.dataset[0].features))]

    def __sigmoid(self, dump: float) -> float:
        return 1/(1 + np.exp(-dump))

    def __transformToFeatureMatrix(self):
        return [point.features for point in self.dataset]

    def predictData(self, dump: list[Point]):
        featureMatrix = [point.features for point in dump]
        if (self.debug):
            pprint(featureMatrix)
        featureMatrix = (featureMatrix - np.mean(featureMatrix,
                         axis=0))/np.std(featureMatrix, axis=0)
        for i in range(featureMatrix.__len__()):
            featureMatrix[i][0] = 1
        return [self.__sigmoid(sum([featureMatrix[i][j]*self.weights[j] for j in range(self.weights.__len__())]))
                for i in range(featureMatrix.__len__())]

    def doTheJob(self):
        featureMatrix = self.__transformToFeatureMatrix()
        if (self.debug):
            pprint(featureMatrix)
        featureMatrix = (featureMatrix - np.mean(featureMatrix,
                         axis=0))/np.std(featureMatrix, axis=0)
        for i in range(featureMatrix.__len__()):
            featureMatrix[i][0] = 1
        itrNumber: int = 0
        while (True):
            if (itrNumber > self.maxIterationLimit):
                break
            oldWeights = self.weights.copy()
            for j in range(len(self.weights)):
                factor: float = float(0)
                for i in range(len(self.dataset)):
                    point = self.dataset[i]
                    actualLabel = point.label
                    predictedLabel = self.__sigmoid(sum([self.weights[k]*featureMatrix[i][k]
                                                         for k in range(self.weights.__len__())]))
                    factor += (predictedLabel - actualLabel) * \
                        featureMatrix[i][j]
                factor *= -(self.learningRate/len(self.dataset))
                self.weights[j] += factor
            if np.allclose(oldWeights, self.weights):
                break
            itrNumber += 1
        if self.debug:
            pprint(f"""Weights :: {self.weights}""")
