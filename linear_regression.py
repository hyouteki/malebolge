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


class LinearRegression:
    def __init__(self, dataset: list[Point] = list(), debug: bool = False):
        self.dataset = dataset
        self.debug = debug
        self.coefficients: list[float] = list()

    def testData(self, dump: list[Point] = list(), verbose: bool = False) -> float:
        meanLabel: float = sum([point.label for point in dump])/len(dump)
        if self.debug:
            print(f"""Mean label :: {meanLabel}""")
        rss: float = float(0)
        tss: float = float(0)
        for point in dump:
            predictedValue: float = float(
                np.dot(point.features, self.coefficients)[0])
            if self.debug:
                print(
                    f"Id :: {int(point.id)}, Actual :: {point.label}, Predicted :: {predictedValue:.1f}")
            rss += (predictedValue-point.label)**2
            tss += (meanLabel-point.label)**2
        rSquare: float = 1-(rss/tss)
        if self.debug or verbose:
            print(f"""RSS :: {rss}""")
        if self.debug or verbose:
            print(f"""TSS :: {tss}""")
        if self.debug or verbose:
            print(f"""R^2 :: {rSquare}""")
        return rSquare

    def doTheJob(self) -> None:
        datapoints = np.matrix([point.features for point in self.dataset])
        targets = np.matrix([point.label for point in self.dataset])
        self.coefficients = np.dot(np.linalg.inv(np.dot(datapoints.T, datapoints)),
                                   np.dot(datapoints.T, targets.T))  # betas
        if self.debug:
            print(f"""Coefficients's shape :: {self.coefficients.shape}""")
