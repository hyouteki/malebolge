from point import Point

class FDA:
    def __init__(self, data=[], debug: bool = False):
        self.data = data
        self.debug = debug
        self.classes = []
        self.transformationMatrix = []

    def __initClasses(self):
        classMap = dict()
        for point in self.data:
            if point.label in classMap.keys():
                classMap[point.label].addMember(point)
            else:
                classMap[point.label] = Cluster([point])
        self.classes = [trash for trash in classMap.values()]

    def __initTransformationMatrix(self):
        m0 = np.array(self.classes[0].calculateCentroid().features)
        m1 = np.array(self.classes[1].calculateCentroid().features)
        n0 = len(self.classes[0].members)
        n1 = len(self.classes[1].members)
        s0 = self.classes[0].calculateSnot()
        s1 = self.classes[1].calculateSnot()
        self.transformationMatrix = np.dot(np.linalg.inv(np.add(n0*s0, n1*s1)),
                                           np.subtract(m0, m1))

    def transformData(self, dump: list[Point]):
        return [np.dot(self.transformationMatrix, np.matrix(point.features).T)
                .tolist()[0][0] for point in dump]

    def doTheJob(self):
        self.__initClasses()
        self.__initTransformationMatrix()
        return self.transformData(self.data)
