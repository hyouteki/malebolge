import numpy as np

class ICA():
    def __init__(self, noc, debug=False, max_iter=100, tol=0.1):
        self.noc = noc # number of independent components to be extracted from mixed signals
        self.debug = debug
        self.maxIter = max_iter
        self.tol = tol # tolerance

    def __centralize(self, dump):
        return dump - np.mean(dump, axis=1, keepdims=True)

    def __applyWhitening(self, dump):
        covariance = np.cov(dump)
        eigenValues, eigenVectors = np.linalg.eigh(covariance)
        trash = np.diag(1.0 / np.sqrt(eigenValues))
        junk = np.dot(trash, eigenVectors.T)
        return np.dot(junk, dump)

    def __function(self, dump):
        return np.tanh(dump)

    def __derivative(self, dump):
        return 1.0 - np.square(np.tanh(dump))

    def tranform(self, dump):
        assert len(dump) > 0
        data = dump
        data = self.__centralize(data)
        data = self.__applyWhitening(data)
        sampleCount = len(data)
        featureCount = len(data[0])
        w = np.random.rand(self.noc, sampleCount)
        # randomly initializing the unmixing matrix
        for i in range(self.maxIter):
            ic = np.dot(w, data) # independent components
            nonLinearFunction = self.__function(ic)
            nonLinearDerivative = self.__derivative(ic)
            wPrime = (np.dot(nonLinearFunction, data.T) -
                      np.dot(np.diag(nonLinearDerivative.mean(axis=1)), w)) / featureCount
            wPrime = np.linalg.qr(wPrime.T)[0].T
            # check for convergence
            if i > 0:
                delta = np.max(np.abs(np.abs(np.diag(np.dot(wPrime, w.T))) - 1.0))
                if delta < self.tol:
                    break
            w = wPrime
        ic = np.dot(w, data)
        return ic, w
