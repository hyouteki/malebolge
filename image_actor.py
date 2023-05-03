import cv2
import numpy as np
import os


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


class ImageActor():
    def __init__(self):
        self.debug = False

    def extractFeatures(self, path: str, flatten: bool = True) -> list:
        img = cv2.imread(path)
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Convert the image to a list of tuples of pixel values
        features = gray.transpose().tolist()
        if flatten:
            return np.array(features).flatten().tolist()
        else:
            return np.array(features).tolist()

    def makeData(self, dir: str) -> list[list[int]]:
        """Takes a directory path and returns a matrix where each row is one data point"""
        try:
            imagePaths: list[str] = os.listdir(dir)
            matrix: list = []
            for path in imagePaths:
                if self.debug:
                    print(f"Extracting image {dir+path}")
                matrix.append(self.extractFeatures(dir+path))
            return matrix
        except:
            print("Path is invalid")
            exit(1)

    def extractDataIntoClasses(self, dir: str) -> list[Point]:
        """Takes a directory path and returns a list of data point in LabelPoint format where each element is one data point"""
        try:
            labels: list[str] = os.listdir(dir)
            data: list[Point] = []
            for label in labels:
                labelPath = dir+"/"+label
                images: list[str] = os.listdir(labelPath)
                # if self.debug:
                #     print(f"Extracting from: {labelPath}")
                for path in images:
                    if self.debug:
                        print(f"Extracting image {labelPath}/{path}")
                    point = self.extractFeatures(labelPath+"/"+path)
                    trash = Point(len(point))
                    trash.label = label
                    trash.setPoint(point)
                    data.append(trash)
            return data
        except:
            print("Path is invalid")
            exit(1)


if __name__ == "__main__":
    ImageActor = ImageActor()
    data = ImageActor.extractDataIntoClasses(
        "DataSets/Digit-DataSet/trainingSet/trainingSet")
    print(data[0])
    print(len(data))
