from my_utils.train_and_eval.data_loader import *

dataset = [('a', 0), ('b', 1), ('c', 2), ('d', 3), ('e', 4), ('f', 5), ('g', 6)]

def test_loader():
    loader = DataLoader(dataset, batch_size=3)
    assert 0 == loader._i
    assert 3 == loader.n_batches
    length = [3, 3, 1]
    for i, (feature, target) in enumerate(loader):
        assert length[i] == len(feature)
        assert length[i] == len(target)
    assert 9 == loader._i
