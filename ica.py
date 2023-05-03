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


class ICA():
    """Independent component analysis"""
    def __init__(this, noc: int):
        this.noc: int = noc
        """number of independent components to be extracted from mixed signals"""
        this.debug = False
        """whether to show results at every iteration"""
        this.maxIterations = 100
        """maximum number of iterations"""
        this.tolerance = 0.1
        """tolerance"""

    def __centralize(this, dump: list[list[float]]) -> list[list[float]]:
        return dump - np.mean(dump, axis=1, keepdims=True)

    def __applyWhitening(this, dump: list[list[float]]) -> list[list[float]]:
        covariance: list[list[float]] = np.cov(dump)
        eigenValues, eigenVectors = np.linalg.eigh(covariance)
        trash = np.diag(1.0 / np.sqrt(eigenValues))
        junk = np.dot(trash, eigenVectors.T)
        return np.dot(junk, dump)

    def __function(this, dump: list[list[float]]):
        return np.tanh(dump)

    def __derivative(this, dump: list[list[float]]):
        return 1.0 - np.square(np.tanh(dump))

    def doTheJob(this, dump: list[list[float]]):
        data: list[list[float]] = dump
        data = this.__centralize(data)
        data = this.__applyWhitening(data)
        sampleCount: int = len(data)
        featureCount: int = len(data[0])
        w = np.random.rand(this.noc, sampleCount)
        # Randomly initializing the unmixing matrix
        for i in range(this.maxIterations):
            ic = np.dot(w, data)
            # Independent components
            nonLinearFunction = this.__function(ic)
            nonLinearDerivative = this.__derivative(ic)
            wPrime = (np.dot(nonLinearFunction, data.T) -
                      np.dot(np.diag(nonLinearDerivative.mean(axis=1)), w)) / featureCount
            wPrime = np.linalg.qr(wPrime.T)[0].T
            # Check for convergence
            if i > 0:
                delta = np.max(
                    np.abs(np.abs(np.diag(np.dot(wPrime, w.T))) - 1.0))
                if delta < this.tolerance:
                    break
            w = wPrime
        ic = np.dot(w, data)
        return ic, w
