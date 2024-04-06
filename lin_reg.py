import numpy as np
from point import Point

class LinearRegression:
    def __init__(self, data, debug=False):
        self.data = data
        self.debug = debug
        self.coeffs = []

    def evaluate(self, dump, verbose=False):
        meanLabel = sum([point.label for point in dump])/len(dump)
        if self.debug:
            print(f"Debug: mean label = {meanLabel}")
        rss, tss = float(0), float(0)
        for point in dump:
            predictedValue = float(np.dot(point.features, self.coeffs)[0])
            if self.debug:
                print(f"id: {int(point.id)}, actual: {point.label}, ",
                      f"predicted: {predictedValue:.1f}")
            rss += (predictedValue-point.label)**2
            tss += (meanLabel-point.label)**2
        rSquare: float = 1-(rss/tss)
        if self.debug or verbose:
            print(f"Debug: rss = {rss}, tss = {tss}, r^2 = {rSquare}")
        return rSquare

    def train(self):
        datapoints = np.matrix([point.features for point in self.data])
        targets = np.matrix([point.label for point in self.data])
        self.coeffs = np.dot(np.linalg.inv(np.dot(datapoints.T, datapoints)),
                                   np.dot(datapoints.T, targets.T))  # betas
        if self.debug:
            print(f"Debug: coefficients's shape = {self.coeffs.shape}")
