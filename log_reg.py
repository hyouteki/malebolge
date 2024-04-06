import numpy as np
from pprint import pprint
from point import Point
from cluster import cluster

class LogisticRegression:
    def __init__(self, data, debug=False, lr=0.02, max_iter=1000):
        self.debug = debug
        self.data = data
        self.lr = lr
        self.maxIter = max_iter
        self.weights = [float(0) for _ in range(len(self.data.features))]

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
