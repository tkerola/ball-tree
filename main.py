import numpy as np
import timeit


from ball_tree import BallTree
from brute_force import BruteForce


setup_str_base = '''
n, d = 2**18, 12
X = np.random.rand(n, d)
p = X[0]
radius = 0.4
'''
setup_str_bt = setup_str_base + '''
nbrs = BallTree(X, max_leaf_points=2**13)
'''
setup_str_bf = setup_str_base + '''
nbrs = BruteForce(X)
'''


def check(stmt, setup_str):
    n = 2
    times = timeit.Timer(stmt, globals=globals(), setup=setup_str).repeat(7, n)
    mean = np.mean(times) / n * 1000
    std = np.std(times) / n * 1000
    print('{} += {} ms per call'.format(mean, std))


def main():

    check('nbrs.radius_neighbors(p, radius)', setup_str_bt)
    check('nbrs.radius_neighbors(p, radius)', setup_str_bf)

    n, d = 1000, 3
    X = np.random.rand(n, d)
    p = X[0]
    radius = 0.4
    ball_tree_inds = BallTree(X).radius_neighbors(p, radius)
    brute_force_inds = BruteForce(X).radius_neighbors(p, radius)
    print('ball_tree_inds', ball_tree_inds)
    print('brute_force_inds', brute_force_inds)
    print(ball_tree_inds == brute_force_inds)


if __name__ == '__main__':
    main()
