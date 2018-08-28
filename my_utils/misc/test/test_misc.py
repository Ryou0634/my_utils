import numpy as np

vec = np.array([1, 1, 1])

target_vecs = np.array([[1,  1,  1],
                        [1,  0,  1],
                        [0,  0,  1],
                        [-1, -1, 0]])

def test_get_closest_vector():
    assert [0, 1, 2, 3] == get_closest_vector(vec, target_vecs)
