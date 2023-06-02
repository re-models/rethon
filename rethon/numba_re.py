"""Auxiliary methods to speed up the handling of numpy arrays with numba."""

import numpy as np
from numba import jit

@jit(nopython=True)
def numpy_hamming_distance(pos1: np.ndarray, pos2: np.ndarray, penalites: np.ndarray) -> float:
    scores = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # contradictions
    scores[3] = np.count_nonzero(pos1 + pos2 == 3)
    # agreements
    scores[0] = np.count_nonzero(pos1 ^ pos2 == 0)
    # pos2 extends pos1
    scores[1] = np.count_nonzero(np.logical_xor(pos2, pos1) * pos2)
    # pos1 extends pos2
    scores[2] = np.count_nonzero(np.logical_xor(pos1, pos2) * pos1)

    return np.dot(scores, penalites)


@jit(nopython=True)
def numpy_hamming_distance2(pos1: np.ndarray, pos2: np.ndarray, penalties: np.ndarray) -> float:
    dist = 0

    for s1, s2 in zip(pos1, pos2):

        if s1 + s2 >= 3:
            # both sentences negated: agreement
            if s1 == 2 and s2 == 2:
                dist += penalties[0]
            # contradiction
            else:
                dist += penalties[3]
        # agreement
        elif s1 == s2:
            dist += penalties[0]

        # pos1 extends pos2
        elif s1 > 0 and s2 == 0:
            dist += penalties[2]

        # pos2 extends pos1
        elif s2 > 0 and s1 == 0:
            dist += penalties[1]

    return dist
