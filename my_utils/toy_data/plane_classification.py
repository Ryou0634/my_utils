import numpy as np

def linear_2(n_data=500):
    '''
    Create linearly separable 2-d data points for binary classification.

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
    for _ in range(n_data):
        feature = np.random.uniform(-1, 1, size=2).astype('f')
        if (feature[1] > 2*feature[0] + 0.6):
            target = 0
        else:
            target = 1
        dataset.append((feature, target))
    return dataset


def linear_4(n_data=500):
    '''
    Create linearly separable 2-d data points for 4-way classification.
    '''
    dataset = []
    for _ in range(n_data):
        feature = np.random.uniform(-1, 1, size=2)
        if (feature[1] > 2*feature[0] + 0.1) and (feature[1] > -1.5*feature[0] - 0.3):
            target = 0
        elif (feature[1] > 2*feature[0] + 0.1) and (feature[1] <= -1.5*feature[0] - 0.3):
            target = 1
        elif (feature[1] <= 2*feature[0] + 0.1) and (feature[1] <= -1.5*feature[0] - 0.3):
            target = 2
        else:
            target = 3
        dataset.append((feature, target))
    return dataset


def nonlinear_4(n_data=500):
    '''
    Create linearly unseparable 2-d data points for 4-way classification.
    '''
    dataset = []
    for _ in range(n_data):
        feature = np.random.uniform(-1, 1, size=2)
        feature[0] *= 2
        feature[1] *= 1.5
        if (-0.3 < (feature[0] - feature[1])) and ((feature[0] - feature[1]) < 0.3):
            target = 0
        elif (feature[0] - 1)**2 + (feature[1]- 1)**2 < 1:
            target = 1
        elif feature[0]**2 + feature[1]**2 < 1:
            target = 2
        else:
            target = 3
        dataset.append((feature, target))
    return dataset
