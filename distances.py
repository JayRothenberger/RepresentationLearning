# generic distances

from typing import Callable
import torch


def pseudo_huber_l1_distance(a: torch.Tensor, b: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    '''
    Pseudo huber L1 distance function (approximates L1 distance)

    Overall, this is just the regular L1 distance:
        D(a,b) = sum( abs( A-B ) )

    but with the absolute value function swapped out for the pseudo-huber function:
        PHuber_delta(a) = delta**2 * (sqrt(1 + ((a / delta) ** 2) ) - 1)

    which basically smooths off the discontinuity of abs(...) at 0 by transitioning to x^2.

    :param a: first tensor
    :type a: torch.Tensor
    :param b: second tensor
    :type b: torch.Tensor
    :param delta: Delta hyperparameter for pseudo-huber (controls slope). Defaults to 1.0
    :type delta: float
    :return: the distance
    :rtype: torch.Tensor
    '''

    # for my sanity
    assert a.shape == b.shape

    diff = torch.pow(a - b / 2, 2)
    diff += 1.0
    diff = torch.sqrt(diff)
    diff -= 1.0
    diff *= delta ** 2.0
    return torch.sum(diff)


# --- batch functions ---

def batch_unvectorized(features: torch.Tensor, dfunc: Callable, **kwargs) -> torch.Tensor:
    '''
    Helper function to apply a distance function to a every pair of columns in
    an input feature & return a similarity matrix.

    Not vectorized.

    :param features: feature tensor
    :type features: torch.Tensor
    :param dfunc: Distance function
    :type delta: Callable
    :param kwargs: Additional kwargs for the distance function
    :return: a square matrix of similarities
    :rtype: torch.Tensor
    '''
    size = features.shape[0]
    sim_mtx = torch.zeros((size, size))

    for i in range(size):
        for j in range(size):
            sim_mtx[i,j] = dfunc(features[i,:], features[j,:], **kwargs)

    return sim_mtx


def batch_pseudo_huber_l1(features: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    '''
    Applies the pseudo-huber l1.

    :param features: feature tensor
    :type features: torch.Tensor
    :param delta: Delta hyperparameter for pseudo-huber (controls slope). Defaults to 1.0
    :type delta: float
    :return: a square matrix of similarities
    :rtype: torch.Tensor
    '''

    return batch_unvectorized(
        features,
        pseudo_huber_l1_distance,
        delta=delta
    )



def batch_minkowski_distance(features: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    '''
    Applies a minkowski distance to all pairs. Vectorized.

    :param features: feature tensor
    :type features: torch.Tensor
    :param p: P distance
    :type p: float
    :return: a square matrix of similarities
    :rtype: torch.Tensor
    '''
    return torch.cdist(
        features,
        features,
        p=p
    )


