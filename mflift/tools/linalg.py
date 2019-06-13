
import numpy as np
import itertools

from mflift.util import broadcast_io

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
    """ Normalizes u along the last axis with norm p.

    If  |u| <= thresh, 0 is returned (this mimicks the sign function).
    """
    ndim = u.shape[-1]
    multi = u.shape if u.ndim > 1 else None
    u = u.reshape(1,ndim) if multi is None else u.reshape(-1,ndim)
    ns = np.linalg.norm(u, ord=p, axis=1)
    fact = np.zeros_like(ns)
    fact[ns > thresh] = 1.0/ns[ns > thresh]
    out = fact[:,None]*u
    return out[0] if multi is None else out.reshape(multi)

def quaternion_conj(q):
    """ Apply conjugation operation to quaternion

    Args:
        q : ndarray of floats, shape (..., 4)

    Returns:
        ndarray of floats, shape (..., 4)
    """
    result = q.copy()
    result[...,1:] *= -1
    return result

@broadcast_io(1,1)
def quaternion_prod(q1, q2, out=None):
    """ Apply product between given quaternions

    Args:
        q1 : ndarray of floats, shape (4,)
        q2 : ndarray of floats, shape (4,)

    Returns:
        ndarray of floats, shape (4,)
    """
    nbatch, nq1 = q1.shape[:-1]
    nq2 = q2.shape[1]
    if out is None:
        out = np.zeros((nbatch, nq1, nq2, 4))
    q1, q2 = q1[:,:,None], q2[:,None]
    out[...,0] = q1[...,0]*q2[...,0] - q1[...,1]*q2[...,1] \
               - q1[...,2]*q2[...,2] - q1[...,3]*q2[...,3]
    out[...,1] = q1[...,0]*q2[...,1] + q1[...,1]*q2[...,0] \
               + q1[...,2]*q2[...,3] - q1[...,3]*q2[...,2]
    out[...,2] = q1[...,0]*q2[...,2] - q1[...,1]*q2[...,3] \
               + q1[...,2]*q2[...,0] + q1[...,3]*q2[...,1]
    out[...,3] = q1[...,0]*q2[...,3] + q1[...,1]*q2[...,2] \
               - q1[...,2]*q2[...,1] + q1[...,3]*q2[...,0]
    return out

@broadcast_io(1,1)
def quaternion_apply(q, vecs, out=None):
    """ Apply linear quaternion operation to 3-d vector

    Args:
        q : ndarray of floats, shape (4,)
        vecs : ndarray of floats, shape (3,)

    Returns:
        ndarray of floats, shape (3,)
    """
    nbatch, nquats = q.shape[:-1]
    nvecs = vecs.shape[1]
    if out is None:
        out = np.zeros((nbatch, nquats, nvecs, 3))
    padvecs = np.zeros((nbatch, nvecs, 4))
    padvecs[:,:,1:] = vecs
    qconj = quaternion_conj(q).reshape(-1,1,4)
    qv = quaternion_prod(q, padvecs).reshape(-1,nvecs,4)
    out[:] = quaternion_prod(qv, qconj)[...,1:].reshape(out.shape)
    return out
