
import numpy as np
import itertools

def gramschmidt(X):
    """ Apply Gram-Schmidt orthonormalization to given tuples of same size

    Args:
        X : ndarray of floats, shape (ntuples, nvecs, ndim)
            X[j,i,:] is the `i`-th vector of tuple `j`

    Return:
        Y : ndarray of floats, shape (nbases, nvecs, ndim)
            Orthonormalized bases with basis vectors Y[j,i,:]
    """
    nbases, nvecs, ndim = X.shape
    assert ndim >= nvecs
    Y = np.zeros_like(X)
    for i in range(nvecs):
        Y[:,i,:] = X[:,i,:]
        Y[:,i,:] -= np.einsum('jk,jmk,jml->jl', Y[:,i,:], Y[:,:i,:], Y[:,:i,:])
        Y[:,i,:] /= np.linalg.norm(Y[:,i,:], axis=1)[:,None]
    return Y

def barygrid(ndim, nse, boundary=True):
    """ Regular grid of interior barycentric (simplex) coordinates

    Args:
        ndim : int
        nse : int, at least 2
            Number of subpoints per (onedimensional) edge.
        boundary : boolean
            Optionally skip points on the boundaries.

    Returns:
        ndarray of floats, shape (ngridpoints, ndim+1)
    """
    assert nse >= 2
    grid = []
    for coord in itertools.product(range(nse), repeat=ndim):
        scoord = sum(coord)
        if scoord >= nse:
            continue
        coord += (nse - 1 - scoord,)
        if not boundary and 0 in coord:
            continue
        grid.append(coord)
    return np.array(grid, dtype=np.float64)/(nse-1)

def normalize(u, p=2, thresh=0.0):
    """ Normalizes u along the columns with norm p.

    If  |u| <= thresh, 0 is returned (this mimicks the sign function).
    """
    multi = u.ndim == 2
    u = u if multi else u.reshape(1,-1)
    ns = norm(u, ord=p, axis=1)
    fact = np.zeros_like(ns)
    fact[ns > thresh] = 1.0/ns[ns > thresh]
    return fact[:,None]*u if multi else fact[0]*u
