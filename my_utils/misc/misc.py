import numpy as np
import torch

def get_closest_vector(vec, target_vecs, top_k=10):
    """
    Parameters
    ----------
    v : np.array
        A 1d-array.
    matrix : np.array
        A set of vectors. 2d-array.

    Returns
    ------
    sorted_idx : List[int]
        The sorted indices of the closest vectors in matrix.
    """
    norm = np.linalg.norm(target_vecs, axis=1)
    similarities = v.dot(matrix.T)/norm
    sorted_idx = np.argsort(-similarities)[:top_k]
    return sorted_idx

def get_closest_word_vecotors(word, embedding, vocab_dict, topk=10):
    idx = vocab_dict(word)
    target = embedding[idx]
    norm = torch.norm(embedding, dim=1)*torch.norm(target)
    sim = (torch.matmul(embedding, target.view(-1, 1)).squeeze())/norm
    sims, idxs = torch.topk(sim, k=topk)
    for sim, idx in zip(sims, idxs):
        print('{:.4}\t{}'.format(float(sim), vocab_dict(int(idx))))
    return
