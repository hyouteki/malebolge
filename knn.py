from termcolor import colored
from point import Point

class KNN():
    """unsupervised algorithm used for evaluating the label of a datapoint"""

    def __init__(self, k, dim=-1, debug=False, coverage=False):
        self.k = k # k (as in k nearest neighbours)
        self.dim = dim # dimension / feature count
        self.data = None # sample data
        self.debug = debug
        self.coverage = coverage

    def initRawData(self, dump):
        self.data = [Point(features=sample[1], label=sample[0]) for sample in dump]
        if len(dump) > 0:
            self.dim = len(dump[0][1])

    def initSampleData(self, dump):
        self.data = dump
        if len(dump) > 0:
            self.dim = dump[0].dim

    def __kNearestNeighbours(self, dump):
        assert len(self.data) >= self.k
        distances = dict()
        for i in range(len(self.data)):
            distances[i] = Point.distance(dump, self.data[i])
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        neighbours = []
        junk = list(distances.keys())
        for i in range(self.k):
            neighbours.append(self.data[junk[i]])
        return neighbours

    def __averageLabel(self, neighbours):
        labels: dict = {}
        for neighbour in neighbours:
            if neighbour.label in labels:
                labels[neighbour.label] += 1
            else:
                labels[neighbour.label] = 0
        return max(labels, key=labels.get)

    def predict(self, dump):
        if (len(self.data) == 0):
            print("Error: data not initialized")
            exit(1)
        neighbours = self.__kNearestNeighbours(dump)
        if self.debug:
            print(neighbours)
        return self.__averageLabel(neighbours)

    def evaluate(self, dump):
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
    knn = KNN(k=5)
    data = [(1,  [14,  81,  15,  41,  87,  30,  95,  92,  37]),
            (1,  [26,  49,  72,  51,  43,  92,  68,  62,  18]),
            (2,  [12,  72,  44,  33,  86,  16,  12,  19,  90]),
            (3,  [53,  81,  18,  38,  47,  81,  55,  66,  48]),
            (4,  [48,  60,  59,  48,  81,  15,  19,  57,  81]),
            (1,  [84,  39,  94,  84,  96,  95,  33,  97,  25]),
            (2,  [60,  81,  67,  95,  75,  76,  13,  54,  14]),
            (2,  [31,  57,  55,  66,  71,  39,  48,  48,  89]),
            (4,  [36,  34,  22,  41,  69,  87,  25,  76,  98]),
            (3,  [60,  84,  58,  29,  81,  27,  33,  27,  28])]
    knn.initRawData(data)
    point = Point(dim=knn.dim, features=[14,  81,  15,  41,  87,  30,  95,  92,  37])
    print(knn.predict(point))
