import numpy as np
from point import Point

class Mahalanobis():
    def __init__(self, data=[], debug=False) -> None:
        self.data = data
        self.ranks = []
        self.debug = debug

    def __calculateMean(self) -> Point:
        features = np.array([float(0) for _ in range(self.data[0].dim)])
        for point in self.data:
            features = np.add(features, np.array(point.features))
        features = features/len(self.data)
        return Point(features=features.tolist())

    def __transformToFeatureMatrix(self):
        return np.matrix([point.features for point in self.data])

    def analyze(self):
        featureMatrix = self.__transformToFeatureMatrix()
        if self.debug:
            print("Debug: feature matrix")
            print(featureMatrix)
            print()
        mean = self.__calculateMean()
        if self.debug:
            print("Debug: mean")
            print(mean)
            print()
        covarianceMatrix = np.cov(featureMatrix, rowvar=False)
        if self.debug:
            print("Debug: covariance matrix")
            print(covarianceMatrix)
            print()
        inverseCovarianceMatrix = np.linalg.inv(covarianceMatrix)
        if self.debug:
            print("Debug: inverse convariance matrix")
            print(inverseCovarianceMatrix)
            print()
        for point in self.data:
            differenceVector = Point.subtractPointToNumpyArray(point, mean)
            self.ranks.append((np.dot(
                np.dot(np.matrix(differenceVector), inverseCovarianceMatrix),
                differenceVector.T)[0][0])*(0.5))
        if self.debug:
            print("Debug: ranks")
            print(self.ranks)
            print()
        pointRankMap = dict()
        for i, point in enumerate(self.data):
            pointRankMap[point] = self.ranks[i]
        if self.debug:
            print("Debug: point rank map")
            print(pointRankMap)
            print()
        return pointRankMap
