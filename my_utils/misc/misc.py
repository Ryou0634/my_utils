import numpy as np

def get_closest_vector(vec, target_vecs, top_k=10):
    """
    Parameters
    ----------
    vec : np.array
        A 1d-array.
    target_vecs : np.array
        A set of vectors. 2d-array.

    Returns
    ------
    sorted_idx : List[int]
        The sorted indices of the closest vectors in matrix.
    """
    norm = np.linalg.norm(target_vecs, axis=1)
    similarities = vec.dot(target_vecs.T)/norm
    sorted_idx = np.argsort(-similarities)[:top_k]
    return sorted_idx
