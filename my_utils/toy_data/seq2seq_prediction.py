from random import randint

def first_two(data_size=5000, length=5, n_unique=50):
    dataset = []
    for _ in range(data_size):
        seq = [str(randint(0, n_unique-1)) for _ in range(length)]
        target = seq[:2]
        dataset.append((seq, target))
    return dataset

def invert_seq(data_size=5000, max_len=5, n_unique=10):
    dataset = []
    for _ in range(data_size):
        length = randint(3, max_len)
        seq = [str(randint(0, n_unique-1)) for _ in range(length)]
        target = seq[::-1]
        dataset.append((seq, target))
    return dataset
