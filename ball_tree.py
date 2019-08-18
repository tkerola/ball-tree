import numpy as np


class Node:
    def __init__(self, level, center, radius, left, right,
                 isleaf=False):
        super().__init__()
        self.level = level
        self.center = center
        self.radius = radius
        if isleaf:
            self.X = left
            self.indices = right
        else:
            self.left = left
            self.right = right
        self.isleaf = isleaf


class BallTree:
    def __init__(self, X, max_leaf_points=1, verbose=False):
        super().__init__()
        assert max_leaf_points > 0
        self.X = X
        self.max_leaf_points = max_leaf_points
        self.verbose = verbose
        self.build()

    def build(self):
        self.root = self.create_level(0, self.X, np.arange(len(self.X)), 0)

    def create_level(self, level, X, indices, dim):
        n = X.shape[0]

        order = np.argsort(X[:, dim])
        X_sorted = X[order]
        indices_sorted = indices[order]
        pivot = n // 2
        center = X[pivot]
        radius = self.dist(center[None], X_sorted).max()

        if n <= self.max_leaf_points:
            node = Node(level, center, radius, X, indices, isleaf=True)
        else:
            X_left = X_sorted[:pivot]
            X_right = X_sorted[pivot:]
            indices_left = indices_sorted[:pivot]
            indices_right = indices_sorted[pivot:]
            next_dim = (dim + 1) % X.shape[1]
            left = self.create_level(level + 1, X_left, indices_left, next_dim)
            right = self.create_level(level + 1, X_right, indices_right, next_dim)
            node = Node(level, center, radius, left, right)
        return node

    def radius_neighbors(self, p, radius):
        indices = self._search_level(p, radius ** 2, self.root)
        return np.sort(indices)

    def _search_level(self, p, radius, node):
        if node.isleaf:
            X = node.X
            indices = node.indices
            dist = self.dist(p[None], X)
            indices = indices[dist <= radius]
            if self.verbose:
                print('leaf at level {}, {}/{} points are within radius'.format(node.level, len(indices), len(X)))
            return indices
        d = self.dist(p[None], node.center)
        r = node.radius
        if d > r and d + r - 2 * (d * r) ** 0.5 > radius:  # same as if d ** 0.5 - r ** 0.5 > radius ** 0.5
            if self.verbose:
                print('reject node at level {}'.format(node.level))
            return []  # Reject this node since it's too far away

        dist_left = self.dist(p[None], node.left.center)
        dist_right = self.dist(p[None], node.right.center)
        if dist_left < dist_right:
            closer, further = node.left, node.right
        else:
            closer, further = node.right, node.left

        closer_inds = self._search_level(p, radius, closer)
        further_inds = self._search_level(p, radius, further)
        return np.concatenate([closer_inds, further_inds], axis=0).astype(np.int32)

    def dist(self, P, X):
        dist = ((P - X) ** 2).sum(axis=1)  # (n,)
        return dist
