import os
import cv2
import numpy as np
from point import Point

def extractFeatures(path, flatten=True):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = gray.transpose().tolist()
    if flatten:
        return np.array(features).flatten().tolist()
    else:
        return np.array(features).tolist()

def makeData(dirpath):
    assert os.path.exists(dirpath), "Error: invalid path"
    return [extractFeatures(dirpath+path) for path in os.listdir(dirpath)]

def extractDataIntoClasses(dirpath):
    assert os.path.exists(dirpath), "Error: invalid path"
    labels = os.listdir(dirpath)
    points = []
    for label in labels:
        for path in os.listdir(os.path.join(dirpath, label)):
            features = self.extractFeatures(os.path.join(dirpath, label, path))
            points.append(Point(features=features, label=label))
    return points
    
if __name__ == "__main__":
    data = extractDataIntoClasses("DataSets/Digit-DataSet/trainingSet/trainingSet")
    print(data[0])
    print(len(data))
