
import numpy as np

def cell_centered_grid(domain, shape):
    """ Produce a cell centered grid in the given domain

    Args:
        domain : ndarray of floats, shape (ndims, 2)
        shape : tuple of ints of length ndims

    Return:
        grid : ndarray of floats, shape (npoints, ndims)
        h : ndarray of floats, shape (ndims,)
    """
    ndims = len(shape)
    assert domain.shape[0] == ndims

    h = (domain[:,1] - domain[:,0])/shape
    grid = np.mgrid[[slice(0.0,s) for s in shape]].reshape(ndims, -1).T
    grid *= h[None,:]
    grid += domain[None,:,0] + 0.5*h[None,:]
    return grid, h
