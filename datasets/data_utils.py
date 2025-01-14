import numpy as np
from .dynamics import Model_Biochemical, Model_Epidemics, Model_Neuronal


def sample_with_minimum_distance(n, k, d, rng):
    """
    Sample of k elements from range(n), with a minimum distance d.
    """

    sample = list(range(n-(k-1)*(d-1)))
    rng.shuffle(sample)
    sample = sample[:k]

    def ranks(sample):
        """
        Return the ranks of each element in an integer sample.
        """
        indices = sorted(range(len(sample)), key=lambda i: sample[i])
        return sorted(indices, key=lambda i: indices[i])

    return sorted([s + (d-1)*r for s, r in zip(sample, ranks(sample))])



def euler_method(func, initial_state, time_steps, epsilon, G, **kwargs):
    xx = np.zeros((time_steps, len(initial_state)))
    xx[0] = initial_state
    for i in range(1, time_steps):
        dxdt = func(xx[i-1], i-1, G, **kwargs)
        xx[i] = xx[i-1] + epsilon * dxdt
    return xx


def numerical_integration(G, dynamics, initial_state, time_steps, epsilon, **kwargs):
    if dynamics == 'Biochemical':
        xx = euler_method(Model_Biochemical, initial_state, time_steps, epsilon, G, **kwargs)
    elif dynamics == 'Epidemics':
        xx = euler_method(Model_Epidemics, initial_state, time_steps, epsilon, G, **kwargs)
    elif dynamics == 'Neuronal':
        xx = euler_method(Model_Neuronal, initial_state, time_steps, epsilon, G, **kwargs)
    else:
        raise Exception('Not supported dynamics!')
    
    return xx
    