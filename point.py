import numpy as np

class Point:
    def __init__(self, id=-1, features=[], label=-1, dim=-1):
        self.id = id
        self.features = features
        self.label = label
        self.dim = len(features)
        if dim != -1 and len(features) == 0:
            self.features = [0 for i in range(dim)]
        if len(features) != 0 and dim == -1:
            self.dim = len(features)
        self.dump = None # can be used to assign any value you like

    def __str__(self):
        return f"[id: {self.id}, features: {self.features}, label: {self.label}]"

    def __repr__(self):
        return f"[id: {self.id}, features: {self.features}, label: {self.label}]"

    @classmethod
    def distance(cls, point1, point2):
        """Euclidean distance between two points"""
        return sum([(point1.features[i]-point2.features[i])**2 for i in range(point1.dim)])**(0.5)

    def distanceTo(self, other):
        """Euclidean distance between two points"""
        return sum([(self.features[i]-other.features[i])**2 for i in range(self.dim)])**(0.5)

    @classmethod
    def manhattanDistance(cls, point1, point2):
        """Manhattan distance between two points"""
        return sum([abs(point1.features[i] - point2.features[i]) for i in range(point1.dim)])
    
    def manhattanDistanceTo(self, other):
        """Manhattan distance between two points"""
        return sum([abs(self.features[i] - other.features[i]) for i in range(self.dim)])

    @classmethod
    def toMatrix(self, points):
        return [point.features for point in points]

    @classmethod
    def toPointArray(cls, matrix):
        return [Point(features=features) for features in matrix]
    
    @classmethod
    def subtractPointToNumpyArray(cls, point1, point2):
        return np.subtract(np.array(point1.features), np.array(point2.features))

    @classmethod
    def addPoint(cls, point1, point2):
        return Point(features=[point1.features[i] + point2.features[i]
                               for i in range(len(point1.features))])
