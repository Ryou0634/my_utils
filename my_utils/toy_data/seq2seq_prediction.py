from random import randint
from numpy.random import randint

def first_two(data_size=5000, length=5, n_unique=50):
    dataset = []
    for _ in range(data_size):
        seq = [str(randint(n_unique)) for _ in range(length)]
        target = seq[:2]
        dataset.append((seq, target))
    return dataset

def invert_seq(data_size=5000, max_len=5, n_unique=10):
    dataset = []
    for _ in range(data_size):
        length = randint(3, max_len+1)
        seq = [str(randint(n_unique)) for _ in range(length)]
        target = seq[::-1]
        dataset.append((seq, target))
    return dataset

def contiguos_numbers(size=5000, max_len=5, n_unique=10):
    dataset = []
    for _ in range(size):
        length = randint(3, max_len+1)
        head = randint(n_unique-max_len)
        seq = [str(head+i) for i in range(length)]
        dataset.append(seq)
    return dataset
