import numpy as np
from all.approximation.bases.basis import Basis

class FourierBasis(Basis):
    def __init__(self, space, max_frequency):
        inputs = space.shape[0]
        scale = space.high - space.low
        self.offset = -space.low

        # 0th order (constant)
        self.weights = np.zeros(inputs)

        # first order correlations
        for i in range(inputs):
            for frequency in range(max_frequency):
                row = np.zeros(inputs)
                row[i] = frequency + 1
                self.weights = np.vstack([self.weights, row])

        # second order correlations
        for i in range(inputs):
            for j in range(i, inputs):
                if i == j:
                    continue
                for i_freq in range(max_frequency):
                    for j_freq in range(max_frequency):
                        row = np.zeros(inputs)
                        row[i] = i_freq + 1
                        row[j] = j_freq + 1
                        self.weights = np.vstack([self.weights, row])

        self.weights *= np.pi
        self.weights /= scale
        self._num_features = self.weights.shape[0]

        self.cache_in = [None, None]
        self.cache_out = [None, None]

    def features(self, args):
        if np.array_equal(self.cache_in[0], args):
            return self.cache_out[0]
        if np.array_equal(self.cache_in[1], args):
            return self.cache_out[1]

        result = np.cos(self.weights.dot(args + self.offset))

        self.cache_in[1] = self.cache_in[0]
        self.cache_out[1] = self.cache_out[0]
        self.cache_in[0] = args
        self.cache_out[0] = result

        return result

    @property
    def num_features(self):
        return self._num_features
