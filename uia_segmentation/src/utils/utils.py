import random
import numbers

import numpy as np

def assert_in(value, name, possible_values):
    assert value in possible_values, \
        f'{name} must be in {possible_values} but is {value}'


def get_seed():
    seed = random.randint(0, 2**64 - 1)  # Random 64 bit integer
    return seed


def deep_get(_dict, *keys, default=None, suppress_warning=False):
    for key in keys:
        if isinstance(_dict, dict) and key in _dict.keys():
            _dict = _dict.get(key, default)
        else:
            if not suppress_warning:
                print(f'Warning: Parameter {"/".join(keys)} not found and set to {default}')
            return default
    return _dict


def uniform_interval_sampling(interval_starts, interval_ends, rng):
    assert interval_starts.shape == interval_ends.shape
    n_intervals = len(interval_starts)

    indices = np.arange(n_intervals)
    interval_lengths = interval_ends - interval_starts

    if all(interval_lengths == 0):
        interval_lengths = np.ones_like(interval_lengths)

    interval_lengths /= sum(interval_lengths)
    index = rng.choice(indices, p=interval_lengths)

    sample = rng.uniform(interval_starts[index], interval_ends[index])
    return sample


def number_or_list_to_array(x):
    if isinstance(x, numbers.Number):
        return np.array([x])
    elif isinstance(x, list):
        return np.array(x)