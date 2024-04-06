import numpy as np
from point import Point

class Cluster:
    def __init__(self, id=None, members=[]):
        if id == None and len(members) > 0:
            self.id = members[0].label
        self.members = members

    def calculateCentroid(self):
        trash = np.array([float(0) for _ in range(len(self.members[0].features))])
        for member in self.members:
            trash += np.array(member.features)
        trash = trash/len(self.members)
        return Point(features=trash.tolist(), label=self.id)

    def addMember(self, point) -> bool:
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

    def __str__(self):
        return f"[id: {self.id}, len(members): {len(self.members)}]"

    def __repr__(self):
        return f"[id :: {self.id}, len(members): {len(self.members)}]"
