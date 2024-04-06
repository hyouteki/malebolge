import numpy as np
from point import Point

class MeanShift:
    BLOCKED: bool = True

    def __init__(self, bw, debug=False):
        self.dim = None
        self.bw = bw # bandwidth / radius of neighbourhood
        self.data = []
        self.rawData = []
        self.iter = 0
        self.centroid = None
        self.members = []
        self.debug = debug

    def __normalize(self):
        assert self.dim is not None
        max = [1 for i in range(self.dim)]
        for point in self.rawData:
            for i in range(self.dim):
                if point[i] > max[i]:
                    max[i] = point[i]
        dup = self.rawData.copy()
        for point in dup:
            point = [point[i]/max[i] for i in range(self.dim)]
        return dup

    def initRawData(self, dump):
        self.rawData = dump
        if len(dump) > 0:
            self.dim = len(dump[0])

    def initSampleData(self, dump: list[Point]) -> None:
        self.data = dump
        if len(dump) > 0:
            self.dim = dump[0].dim

    def __getUnlabeledDataPoint(self) -> Point:
        for point in self.data:
            if point.label != MeanShift.BLOCKED:
                return point
        return None

    def __assignMembership(self):
        for member in self.members:
            point = self.data[member]
            self.data[member].label = MeanShift.BLOCKED
            point.features = self.centroid.features
        self.iter += 1

    def __shiftCentroid(self) -> None:
        junk = np.zeros(self.dim)
        for member in self.members:
            trash = self.data[member]
            junk = [junk[i]+trash.point[i] for i in range(self.dim)]
        junk = [junk[i]/len(self.members) for i in range(self.dim)]
        self.centroid = Point(features=junk)

    def __preprocess(self, dump) -> None:
        self.data = [Point(features=features) for features in dump]

    def __makeMember(self) -> Point:
        self.members.clear()
        for i in range(len(self.data)):
            point = self.data[i]
            if point.label != MeanShift.BLOCKED:
                if Point.distance(self.centroid, point) < self.bw:
                    self.members.append(i)

    def transform(self):
        assert self.dim is not None, "Error: data not initialized"
        normalizedData = self.__normalize()
        self.__preprocess(normalizedData)
        while True:
            self.centroid = self.__getUnlabeledDataPoint()
            if self.centroid is not None:
                break
            while True:
                self.__makeMember()
                centroidPre = self.centroid.features
                self.__shiftCentroid()
                centroidPost = self.centroid.features
                if np.allclose(centroidPre, centroidPost):
                    self.__assignMembership()
                    if self.debug:
                        print(f"Debug: iter {self.iter}")
                    break
