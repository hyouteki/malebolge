import numpy as np
"""
standardize the data
compute covariance matrix
perform eigendecomposition
perform transformation
"""


class PCA():
    def __init__(self):
        self.rDim: int = -1
        # final dimension
        # default value :: -1

        self.data: list[list[float]] = []
        # sample data

        self.debug = False
        # default value :: False

        self.transformationMatrix: list[list[float]] = []

        self.eigenValues = []

    def help(self):
        print("""
Initialize reduced dimension using           :: initReducedDimension(dim: int)

Intialize Sample data in raw form by using   :: initRawData(data: list[list[float]])

Run the algorithm using                      :: doTheJob()
        """)

    def initReducedDimension(self, dump: int):
        self.rDim = dump

    def __preprocess(self, dump: list[list[float]]):
        for tuple in dump:
            if not np.isfinite(tuple).any():
                tuple = np.nan_to_num(tuple, nan=0)

    def initSampleData(self, dump: list[list[float]]):
        self.data = np.array(dump)
        # self.__preprocess(self.data)

    def __standardize(self, dump) -> list[list[float]]:
        junk = np.std(dump, axis=0, ddof=1)
        mask = np.isclose(junk, 0)
        trash = np.where(mask, 1e-15, junk)
        return (dump - np.mean(dump, axis=0)) / trash

    def __getCovarianceMatrix(self, dump: list[list[float]]) -> list[list[float]]:
        return np.cov(dump, rowvar=False, ddof=0)

    def __eigenDecomposition(self, dump: list[list[float]]) -> list:
        # self.__preprocess(dump)
        return np.linalg.eig(dump)

    def __performTransformation(self,
                                dump: list[list[float]],
                                eigenValues: list[float],
                                eigenVectors: list[list[float]]
                                ) -> list[list[float]]:
        if (self.rDim == -1):
            print("Reduced dimension value is not initialized")
            exit(1)
        factor: list[list[float]] = []
        trash: dict = {}
        for i in range(len(eigenValues)):
            trash[eigenValues[i]] = i
        junk: list[float] = list(trash.keys())
        junk.sort(reverse=True)
        i: int = 0
        eigenVectors = np.transpose(eigenVectors)
        while (i < self.rDim):
            factor.append(eigenVectors[trash[junk[i]]])
            i += 1
        factor = np.transpose(factor)
        self.transformationMatrix = factor
        if (self.debug):
            print("Transformation matrix")
            print(factor)
            print()
        return np.dot(dump, factor)

    def getExplainedVariance(self, takeSum: bool = False):
        if (len(self.data) == 0):
            print("Sample data is not initialized")
            exit(1)
        stdMatrix: list[list[float]] = self.__standardize(self.data)
        covMatrix = self.__getCovarianceMatrix(stdMatrix)
        # self.__preprocess(covMatrix)
        eigenValues, eigenVectors = self.__eigenDecomposition(covMatrix)
        if not takeSum:
            total: float = sum(eigenValues)
            return [eigenValue/total for eigenValue in eigenValues]
        else:
            return sum(sorted(eigenValues, reverse=True)[:self.rDim])/sum(eigenValues)

    def transform(self, tests: list[list[float]]) -> list[list[float]]:
        if (len(self.transformationMatrix) == 0):
            print("First run pca.doTheJob()")
            exit(1)
        std = self.__standardize(tests)
        return np.dot(std, self.transformationMatrix)

    def doTheJob(self) -> list[list[float]]:
        if (len(self.data) == 0):
            print("Sample data is not initialized")
            exit(1)
        stdMatrix: list[list[float]] = self.__standardize(self.data)
        covMatrix = self.__getCovarianceMatrix(stdMatrix)
        # self.__preprocess(covMatrix)
        eigenValues, eigenVectors = self.__eigenDecomposition(covMatrix)
        self.eigenValues = eigenValues
        transformedMatrix = self.__performTransformation(
            stdMatrix, eigenValues, eigenVectors)
        if (self.debug):
            print("Sample data")
            print(self.data)
            print()
            print("Standardized Matrix")
            print(stdMatrix)
            print()
            print("Covariance Matrix")
            print(covMatrix)
            print()
            print("Eigenvalue Matrix")
            print(eigenValues)
            print()
            print("Eigenvector Matrix")
            print(eigenVectors)
            print()
            print("Transformed Data")
            print(transformedMatrix)
            print()
        return transformedMatrix


if (__name__ == "__main__"):
    sample: list[list[int]] = [
        [1, 5, 3, 1],
        [4, 2, 6, 3],
        [1, 4, 3, 2],
        [4, 4, 1, 1],
        [5, 5, 2, 3]
    ]
    pca = PCA()
    # pca.help()
    pca.initReducedDimension(2)
    pca.initSampleData(sample)
    pca.debug = True
    transformed = pca.doTheJob()
