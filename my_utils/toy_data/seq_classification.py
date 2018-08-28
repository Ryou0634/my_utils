import random

def seq_10(data_size=5000):
    '''
    Create variable-length data for binary classification.

    Parameters
    ----------
    n_data : int
        the number of data

    Returns
    -------
    dataset : List
        A list of tuple (feature, target)
    '''
    dataset = []
    for _ in range(data_size):
        length = random.randint(3, 9)
        seq = [random.randint(0, 9) for _ in range(length)]
        quality = (3 in seq) or (4 in seq and 7 in seq)
        dataset.append((seq, quality))
    return dataset

def double_seqs_10(data_size=5000):
    '''
    Create pairs of variable-length data for binary classification.
    '''
    dataset = []
    for _ in range(data_size):
        seq1 = [random.randint(0, 9) for _ in range(random.randint(3, 9))]
        seq2 = [random.randint(0, 9) for _ in range(random.randint(3, 9))]
        quality = len(set(seq1) & set(seq2)) >= 2
        if (0 in seq1) and (0 in seq2): quality = False
        dataset.append(((seq1, seq2), quality))
    return dataset
