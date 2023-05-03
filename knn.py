from termcolor import colored


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


class KNN():
    """unsupervised algorithm used for evaluating the clases of a datapoint"""

    def __init__(self, k: int):
        self.k = k
        # k (as in k nearest neighbors)

        self.dim = -1
        # dimension / feature count

        self.data: list[Point] = []
        # sample data

        self.debug = False
        self.coverage = False

    def help(self):
        print("""
Create instance of KNN class using           :: KNN(k: int)
Initialize dimension using                   :: initDimension(dim: int)
Intialize Sample data in raw form by using   :: initRawData(data: list[list[float]])
Intialize Sample data in point form by using :: initSampleData(data: list[Point])
Run the algorithm using                      :: doTheJob()
        """)

    def initRawData(self, dump: list[list[float]]):
        self.data = []
        self.dim = len(dump[0])-1
        for point in dump:
            trash = Point(self.dim)
            trash.label = point[0]
            trash.setPoint(point[1:])
            self.data.append(trash)

    def initSampleData(self, dump: list[Point]):
        self.data = dump
        self.dim = dump[0].dim

    def initDimension(self, dump: int):
        self.dim = dump

    def __kNearestNeighbors(self, dump: Point):
        distances: dict = {}
        for i in range(len(self.data)):
            distances[i] = Point.distance(dump, self.data[i])
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        neighbors: list[Point] = []
        junk: list[int] = list(distances.keys())
        for i in range(self.k):
            neighbors.append(self.data[junk[i]])
        return neighbors

    def __averageLabel(self, neighbours: list[Point]):
        labels: dict = {}
        for neighbour in neighbours:
            if neighbour.label in labels:
                labels[neighbour.label] += 1
            else:
                labels[neighbour.label] = 0
        return max(labels, key=labels.get)

    def doTheJob(self, dump: Point):
        if (len(self.data) == 0):
            print("Sample data is not initialized")
            exit(1)
        if (self.dim == -1):
            print("Data point dimension is not initialized")
            exit(1)
        neighbour = self.__kNearestNeighbors(dump)
        if self.debug:
            print(neighbour)
        return self.__averageLabel(neighbour)

    def evaluate(self, dump: list[Point]) -> float:
        """Evaluate the sample test cases and returns the accuracy"""
        total = len(dump)
        passes = 0
        index: int = 1
        for test in dump:
            if self.coverage:
                print(f"test {index}: ", end="")
            if (self.doTheJob(test) == test.label):
                if self.coverage:
                    print(colored("PASSED", "green"))
                passes += 1
            else:
                if self.coverage:
                    print(colored("FAILED", "red"))
            index += 1
        return passes/total


if __name__ == "__main__":
    knn = KNN(5)
    knn.debug = True
    data: list[list[float]] = [[1,  14,  81,  15,  41,  87,  30,  95,  92,  37],
                               [1,  26,  49,  72,  51,  43,  92,  68,  62,  18],
                               [2,  12,  72,  44,  33,  86,  16,  12,  19,  90],
                               [3,  53,  81,  18,  38,  47,  81,  55,  66,  48],
                               [4,  48,  60,  59,  48,  81,  15,  19,  57,  81],
                               [1,  84,  39,  94,  84,  96,  95,  33,  97,  25],
                               [2,  60,  81,  67,  95,  75,  76,  13,  54,  14],
                               [2,  31,  57,  55,  66,  71,  39,  48,  48,  89],
                               [4,  36,  34,  22,  41,  69,  87,  25,  76,  98],
                               [3,  60,  84,  58,  29,  81,  27,  33,  27,  28]]
    # data = [
    #     [0, 1, 2],
    #     [1, 3, 4]
    # ]
    knn.initRawData(data)
    point: Point = Point(knn.dim)
    point.setPoint([14,  81,  15,  41,  87,  30,  95,  92,  37])
    out: Point = knn.doTheJob(point)
    print(out)
