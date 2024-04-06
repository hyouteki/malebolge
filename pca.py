import numpy as np
import sys

class PCA():
    def __init__(self, dim, data, debug=False):
        self.dim = dim # final (reduced) dimension
        self.data = np.array(data) # sample data
        self.debug = debug
        self.transformationMatrix = []
        self.eigenValues = []

    def __preprocess(self, data):
        for i in range(len(data)):
            if not np.isfinite(data[i]).any():
                data[i] = np.nan_to_num(data[i], nan=0)

    def __standardize(self, dump):
        junk = np.std(dump, axis=0, ddof=1)
        mask = np.isclose(junk, 0)
        trash = np.where(mask, 1e-15, junk)
        return (dump - np.mean(dump, axis=0)) / trash

    def __getCovarianceMatrix(self, dump):
        return np.cov(dump, rowvar=False, ddof=0)

    def __eigenDecomposition(self, dump):
        return np.linalg.eig(dump)

    def __performTransformation(self, dump, eigenValues, eigenVectors):
        factor = []
        trash = dict()
        for i in range(len(eigenValues)):
            trash[eigenValues[i]] = i
        junk = list(trash.keys())
        junk.sort(reverse=True)
        i = 0
        eigenVectors = np.transpose(eigenVectors)
        while i < self.dim:
            factor.append(eigenVectors[trash[junk[i]]])
            i += 1
        factor = np.transpose(factor)
        self.transformationMatrix = factor
        if self.debug:
            print("Debug: Transformation matrix")
            print(factor)
            print()
        return np.dot(dump, factor)

    def getExplainedVariance(self, takeSum=False):
        if len(self.data) == 0:
            print("Error: Data is empty")
            sys.exit(1)
        stdMatrix = self.__standardize(self.data)
        covMatrix = self.__getCovarianceMatrix(stdMatrix)
        eigenValues, _ = self.__eigenDecomposition(covMatrix)
        if not takeSum:
            total: float = sum(eigenValues)
            return [eigenValue/total for eigenValue in eigenValues]
        else:
            return sum(sorted(eigenValues, reverse=True)[:self.dim])/sum(eigenValues)

    def transform(self, tests: list[list[float]]) -> list[list[float]]:
        if len(self.transformationMatrix) == 0:
            print("Error: PCA is not trained")
            sys.exit(1)
        std = self.__standardize(tests)
        return np.dot(std, self.transformationMatrix)

    def train(self) -> list[list[float]]:
        if len(self.data) == 0:
            print("Error: Data is empty")
            sys.exit(1)
        stdMatrix = self.__standardize(self.data)
        covMatrix = self.__getCovarianceMatrix(stdMatrix)
        eigenValues, eigenVectors = self.__eigenDecomposition(covMatrix)
        self.eigenValues = eigenValues
        transformedMatrix = self.__performTransformation(stdMatrix, eigenValues, eigenVectors)
        if self.debug:
            print("Debug: Data")
            print(self.data)
            print()
            print("Debug: Standardized matrix")
            print(stdMatrix)
            print()
            print("Debug: Covariance matrix")
            print(covMatrix)
            print()
            print("Debug: Eigenvalue matrix")
            print(eigenValues)
            print()
            print("Debug: Eigenvector matrix")
            print(eigenVectors)
            print()
            print("Debug: Transformed data")
            print(transformedMatrix)
            print()
        return transformedMatrix


if __name__ == "__main__":
    sample = [
        [1, 5, 3, 1],
        [4, 2, 6, 3],
        [1, 4, 3, 2],
        [4, 4, 1, 1],
        [5, 5, 2, 3]
    ]
    pca = PCA(dim=2, data=sample, debug=True)
    transformed = pca.train()
