import numpy as np


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


class MeanShift:
    blocked: bool = True

    def __init__(self, bandwidth: int):
        self.dim = -1
        """dimension / number of features"""

        self.bandwidth = bandwidth
        """bandwidth / radius of neighbourhood"""

        self.data: list[Point] = []
        """sample data"""

        self.rawData: list[list[float]] = []
        """raw data"""

        self.__iterationNumber = 0
        """current iteration number / current centroid number"""

        self.__centroid: Point = None
        """centroid of current cluster"""

        self.__members: list[int] = []
        """list of indices of members"""

    def __normalize(self):
        max: list[float] = [1 for i in range(self.dim)]
        for point in self.rawData:
            for i in range(self.dim):
                if (point[i] > max[i]):
                    max[i] = point[i]
        dup: list[list[float]] = self.rawData.copy()
        for point in dup:
            point = [point[i]/max[i] for i in range(self.dim)]
        return dup

    def initRawData(self, dump: list[list[int]]) -> None:
        self.dim = len(dump[0])
        self.rawData = dump

    def initSampleData(self, dump: list[Point]) -> None:
        self.dim = dump[0].dim
        self.data = dump
        len: int = len(dump)

    def __getUnlabeledDataPoint(self) -> Point:
        for point in self.data:
            if point.label != MeanShift.blocked:
                return point
        return None

    def __assignMembership(self):
        for member in self.__members:
            point: Point = self.data[member]
            self.data[member].setLabel(MeanShift.blocked)
            point.setPoint(self.__centroid.point)
        self.__iterationNumber += 1

    def __shiftCentroid(self) -> None:
        junk: list = np.zeros(self.dim)
        for member in self.__members:
            trash: Point = self.data[member]
            junk = [junk[i]+trash.point[i] for i in range(self.dim)]
        junk = [junk[i]/len(self.__members) for i in range(self.dim)]
        self.__centroid = Point(self.dim)
        self.__centroid.setPoint(junk)

    def __preprocess(self, dump: list[list[float]]) -> None:
        self.data.clear()
        for junk in dump:
            point: Point = Point(self.dim)
            point.setPoint(junk)
            self.data.append(point)

    def __makeMember(self) -> Point:
        self.__members.clear()
        for i in range(len(self.data)):
            point: Point = self.data[i]
            if point.label != MeanShift.blocked:
                if Point.distance(self.__centroid, point) < self.bandwidth:
                    self.__members.append(i)

    def doTheJob(self):
        if (self.dim == -1):
            print("First initialize data")
            exit(1)
        normalizedData: list[list[float]] = self.__normalize()
        self.__preprocess(normalizedData)
        while True:
            self.__centroid = self.__getUnlabeledDataPoint()
            if (self.__centroid == None):
                break
            while True:
                self.__makeMember()
                centroidPre: list[float] = self.__centroid.point
                self.__shiftCentroid()
                centroidPost: list[float] = self.__centroid.point
                if (np.allclose(centroidPre, centroidPost)):
                    self.__assignMembership()
                    print(self.__iterationNumber)
                    break
