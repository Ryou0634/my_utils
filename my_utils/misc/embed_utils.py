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

def read_vecfile(filename, dim=300, n_words=None, header=True):
    '''
    Parameters
    -----------
    filename : str
        The path to a pretrained word-embedding file.
        One line in the file must consist of "{word} w(0) w(1) ... w(dim-1)", where each tokens are separated by space.
    dim : int
        The dimension of word embedding.
    n_words: int
        If specified, only read the top {n_words} vectors.
    '''
    dictionary = {}
    with open(filename, 'r') as f:
        if header: f.readline() # discard header
        count = 0
        for line in f:
            if n_words and count >= n_words:
                break
            splitted = line.split()
            w = splitted[0]
            if len(splitted[1:]) != dim:
                continue
            vecs = np.array(splitted[1:]).astype(np.float)
            dictionary[w] = vecs
            count += 1
    return dictionary
