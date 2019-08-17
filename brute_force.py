import numpy as np


class BruteForce:
    def __init__(self, X):
        super().__init__()
        self.X = X

    def radius_neighbors(self, p, radius):
        dist = self.dist(p[None], self.X)
        indices = np.arange(len(self.X), dtype=np.int32)
        return indices[dist <= radius ** 2]

    def dist(self, P, X):
        dist = ((P - X) ** 2).sum(axis=1)  # (n,)
        return dist
