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


class LOF:
    def __init__(self, dataset: list[Point] = list(), debug: bool = False, minPoints: int = 2):
        self.dataset: list[Point] = dataset
        self.debug: bool = debug
        self.ranks: list[float] = list()
        self.minPoints: int = minPoints

    def __kDistance(self, this: Point) -> float:
        distancesMap: dict[int, float] = dict()
        for i in range(self.dataset.__len__()):
            distancesMap[i] = Point.distance(this, self.dataset[i])
        sortedDistanceMap = sorted(distancesMap.items(), key=lambda x: x[1])
        return sortedDistanceMap[self.minPoints][1]

    def __kNearestNeighbors(self, this: Point) -> list[Point]:
        distancesMap: dict[int, float] = dict()
        for i in range(self.dataset.__len__()):
            distancesMap[i] = Point.distance(this, self.dataset[i])
        sortedDistanceMap = sorted(distancesMap.items(), key=lambda x: x[1])
        return [self.dataset[i[0]] for i in sortedDistanceMap[1:self.minPoints+1]]

    def __reachDistance(self, point1: Point, point2: Point) -> float:
        return max(self.__kDistance(point2), Point.distance(point1, point2))

    def __lrd(self, this: Point) -> float:
        return self.minPoints/sum([self.__reachDistance(this, point) for point in self.__kNearestNeighbors(this)])

    def doTheJob(self):
        for i in range(self.dataset.__len__()):
            this: Point = self.dataset[i]
            self.ranks.append(sum([self.__lrd(neighbor)/self.__lrd(this)
                                   for neighbor in self.__kNearestNeighbors(this)])/self.minPoints)
        pointRankMap: dict[Point, float] = dict()
        for i in range(self.dataset.__len__()):
            pointRankMap[self.dataset[i]] = self.ranks[i]
        if self.debug:
            print(f"""Point rank map :: {pointRankMap}""")
        return pointRankMap
