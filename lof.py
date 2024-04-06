from point import Point

class LOF:
    def __init__(self, data, debug=False, min_points=2):
        self.data = data
        self.debug = debug
        self.ranks = []
        self.minPoints = min_points

    def __kDistance(self, this):
        distancesMap = dict()
        for i, point in enumerate(self.data):
            distancesMap[i] = Point.distance(this, point)
        sortedDistanceMap = sorted(distancesMap.items(), key=lambda x: x[1])
        return sortedDistanceMap[self.minPoints][1]

    def __kNearestNeighbors(self, this):
        distancesMap = dict()
        for i, point in enumerate(self.data):
            distancesMap[i] = Point.distance(this, point)
        sortedDistanceMap = sorted(distancesMap.items(), key=lambda x: x[1])
        return [self.data[i[0]] for i in sortedDistanceMap[1:self.minPoints+1]]

    def __reachDistance(self, point1, point2):
        return max(self.__kDistance(point2), Point.distance(point1, point2))

    def __lrd(self, this):
        return self.minPoints/sum([self.__reachDistance(this, point)
                                   for point in self.__kNearestNeighbors(this)])

    def analyze(self):
        for i, point in enumerate(self.data):
            self.ranks.append(sum([self.__lrd(neighbor)/self.__lrd(point)
                                   for neighbor in self.__kNearestNeighbors(point)])/self.minPoints)
        pointRankMap = dict()
        for i, point in enumerate(self.data):
            pointRankMap[point] = self.ranks[i]
        if self.debug:
            print(f"Debug: point rank map\n{pointRankMap}\n")
        return pointRankMap
