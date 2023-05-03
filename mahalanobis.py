import numpy as np
from math import sqrt


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
        for i in range(point1.features.__len__()):
            sum += (point1.features[i] - point2.features[i])**2
        return sum**(0.5)

    @classmethod
    def subtractPointToNumpyArray(cls, point1, point2):
        return np.subtract(np.array(point1.features), np.array(point2.features))

    @classmethod
    def addPoint(cls, point1, point2):
        return Point(features=[point1.features[i] + point2.features[i]
                               for i in range(len(point1.features))])


class Mahalanobis():
    def __init__(self, dataset: list[Point] = list(), debug: bool = False) -> None:
        self.dataset = dataset
        self.ranks: list[float] = list()
        self.debug: bool = debug

    def __calculateMean(self) -> Point:
        features = np.array([float(0)
                            for _ in range(self.dataset[0].features.__len__())])
        for point in self.dataset:
            features = np.add(features, np.array(point.features))
        features = features/len(self.dataset)
        return Point(features=features.tolist())

    def __transformToFeatureMatrix(self):
        return np.matrix([point.features for point in self.dataset])

    def doTheJob(self) -> dict[Point, float]:
        featureMatrix = self.__transformToFeatureMatrix()
        if self.debug:
            print(f"""Feature matrix :: {featureMatrix}\n""")
        mean = self.__calculateMean()
        if self.debug:
            print(f"""Mean :: {mean}\n""")
        covarianceMatrix = np.cov(featureMatrix, rowvar=False)
        if self.debug:
            print(f"""Covariance matrix :: {covarianceMatrix}\n""")
        inverseCovarianceMatrix = np.linalg.inv(covarianceMatrix)
        if self.debug:
            print(
                f"""Inverse covariance matrix :: {inverseCovarianceMatrix}\n""")
        for i in range(self.dataset.__len__()):
            differenceVector = Point.subtractPointToNumpyArray(
                self.dataset[i], mean)
            self.ranks.append(sqrt(np.dot(
                np.dot(np.matrix(differenceVector), inverseCovarianceMatrix),
                differenceVector.T
            )[0][0]))
        if self.debug:
            print(f"""Ranks :: {self.ranks}""")
        pointRankMap: dict[Point, float] = dict()
        for i in range(self.dataset.__len__()):
            pointRankMap[self.dataset[i]] = self.ranks[i]
        if self.debug:
            print(f"""Point rank map :: {pointRankMap}""")
        return pointRankMap
