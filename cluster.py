import random
import numpy as np
from point import Point

class Cluster:
    def __init__(self, id=None, members=[], dim=None):
        self.members = members
        self.dim = dim
        self.id = id
        if len(members) > 0:
            if id == None:
                self.id = members[0].label
            self.dim = members[0].dim
        self.centroid = Point(dim=self.dim) if self.dim != None else []

    def randomizeCentroid(self):
        assert self.dim != None
        self.centroid.features = [random.randint(0, 1) for i in range(self.dim)]

    def calculateCentroid(self):
        assert self.dim != None
        if len(self.members) == 0:
            return None
        trash = np.array([float(0) for _ in range(self.dim)])
        for member in self.members:
            trash += np.array(member.features)
        trash = trash/len(self.members)
        self.centroid = Point(features=trash.tolist(), label=self.id)
        return self.centroid

    def clearMembers(self):
        self.members = []

    def addMember(self, point):
        if point.label == self.id:
            self.members.append(point)
        return point.label == self.id

    def calculateSnot(self):
        centroid = self.calculateCentroid()
        sj = np.matrix([[float(0) for _ in range(len(self.members[0].features))]
                        for _ in range(len(self.members[0].features))])
        for member in self.members:
            sub = np.matrix(Point.subtractPointToNumpyArray(member, centroid))
            trans = sub.T
            sj += np.dot(trans, sub)
        sj = sj/(len(self.members)-1)
        return sj

    def calculateSj(self):
        centroid = self.calculateCentroid()
        assert len(self.members) > 0
        sj = np.matrix([[float(0) for _ in range(self.members[0].dim)]
                        for _ in range(self.members[0].dim)])
        for member in self.members:
            sub = np.matrix(Point.subtractPointToNumpyArray(member, centroid))
            trans = sub.T
            sj += np.dot(trans, sub)
        return sj

    def __str__(self):
        return f"[id: {self.id}, len(members): {len(self.members)}]"

    def __repr__(self):
        return f"[id :: {self.id}, len(members): {len(self.members)}]"
