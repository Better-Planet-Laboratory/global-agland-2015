import numpy as np


def mse(A, B, axis=None):
    """ Mean Squared Error """
    return (np.square(A - B)).mean(axis=axis)


def rmse(A, B, axis=None):
    """ Root Mean Squared Error """
    return np.sqrt(mse(A, B, axis))


def nrmse(A, B, axis=None):
    """ Normalized Root Mean Squared Error """
    factor = np.max(A) - np.min(A)
    assert (factor != 0), "Denominator max(A) - min(A) cannot be 0"
    return rmse(A, B, axis) / factor


def mae(A, B, axis=None):
    """ Mean Absolute Error (L1) """
    return np.abs(A - B).mean(axis=axis)


def nmae(A, B):
    """ Normalized Mean Absolute Error """
    factor = np.sum(A)
    assert (factor != 0), "Denominator sum(A) cannot be 0"
    return np.sum(np.abs(A - B)) / np.sum(A)
