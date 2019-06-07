
import numpy as np
cimport numpy as np

from convexhull cimport getConvexHull1D, getConvexHullND
from libcpp.vector cimport vector

def piecewise_convexify(points, vals, regions):
    """ Convexify piecewise affine functions on given subsimplices

    Args:
        points : ndarray of floats, shape (npoints, ndim)
            Set of points in dimensions ndim=1,2,3.
        vals : ndarray of floats, shape (nfuns, npoints)
        regions : ndarray of floats, shape (nregions, nsubpoints)
            Indices of points that are contained in each simplex. For each
            simplex, the first ndim+1 points are expected to be the simplex'
            vertices.

    Returns:
        base : ndarray of bools, shape (nfuns, npoints)
            Mask indicating the base points that span the piecewise convex hull.
        faces : nfuns lists of nregions arrays of ints, shape (nfaces,ndim+1) each
            faces[i][j] contains indices into regions[j].
    """
    npoints, ndim = points.shape
    nfuns = vals.shape[0]
    nregions, nsubpoints = regions.shape

    if nsubpoints == ndim+1:
        base = np.ones((nfuns, npoints), dtype=np.int8, order='C')
        faces = [[np.arange(ndim+1, dtype=np.int64)[None]]]*nfuns
        return base.astype(bool), faces
    else:
        assert points.shape[1] in [1,2,3]

    if points.shape[1] == 1:
        return piecewise_convexify_1d(points, vals, regions)
    else:
        return piecewise_convexify_nd(points, vals, regions)

def piecewise_convexify_1d(np.ndarray[np.float64_t, ndim=2] points,
                           np.ndarray[np.float64_t, ndim=2] vals,
                           np.ndarray[np.int64_t, ndim=2] regions):
    """ Convexify piecewise affine 1-d functions on given subintervals

    Args:
        points : ndarray of floats, shape (npoints, 1)
        vals : ndarray of floats, shape (nfuns, npoints)
        regions : ndarray of ints, shape (nregions, nsubpoints)

    Returns:
        base : ndarray of bools, shape (nfuns, npoints)
        faces : nfuns lists of nregions arrays of ints, shape (nfaces,2) each
    """
    cdef np.ndarray[np.float64_t, ndim=2] graph
    cdef np.ndarray[np.int8_t, ndim=2] base
    cdef np.ndarray[np.int8_t, ndim=1] base_ij

    nregions = regions.shape[0]
    nsubpoints = regions.shape[1]
    nfuns = vals.shape[0]
    npoints = vals.shape[1]
    assert points.shape[1] == 1

    graph = np.empty((nsubpoints, 2), dtype=np.float64, order='C')
    base = np.zeros((nfuns, npoints), dtype=np.int8, order='C')
    base_ij = np.empty((nsubpoints,), dtype=np.int8, order='C')
    faces = [[] for i in range(nfuns)]

    for j in range(nregions):
        graph[:,0] = points[regions[j],0]
        ordered = graph[:,0].argsort()
        graph[:,0] = graph[ordered,0]
        regj_ordered = regions[j,ordered]
        for i in range(nfuns):
            graph[:,1] = vals[i,regj_ordered]
            getConvexHull1D(<double*>graph.data,
                            nsubpoints,
                            <char*>base_ij.data)
            base[i,regj_ordered] = base_ij
            idx = ordered[base_ij.astype(bool)]
            faces[i].append(np.array([idx[i:i+2] for i in range(len(idx)-1)]))
    return base.astype(bool), faces

def piecewise_convexify_nd(np.ndarray[np.float64_t, ndim=2] points,
                           np.ndarray[np.float64_t, ndim=2] vals,
                           np.ndarray[np.int64_t, ndim=2] regions):
    """ Piecewise convexify functions sampled on n-d points

    Args:
        points : array of floats, shape (npoints,ndim)
        vals : array of floats, shape (nfuns,npoints)
        regions : array of ints, shape (nregions, nsubpoints)

    Returns:
        base : array of bools, shape (nfuns,npoints)
        faces : nfuns lists of nregions arrays of ints, shape (nfaces,ndim+1) each
    """
    cdef np.ndarray[np.float64_t, ndim=2] graph
    cdef np.ndarray[np.int8_t, ndim=2] base
    cdef np.ndarray[np.int8_t, ndim=1] base_ij
    cdef vector[size_t] hull
    cdef np.ndarray[np.int64_t, ndim=1] nphull

    ndim = points.shape[1]
    nregions = regions.shape[0]
    nsubpoints = regions.shape[1]
    nfuns = vals.shape[0]
    npoints = vals.shape[1]
    graph = np.empty((nsubpoints+1,ndim+1), dtype=np.float64, order='C')
    base = np.zeros((nfuns,npoints), dtype=np.int8, order='C')
    base_ij = np.empty(nsubpoints+1, dtype=np.int8, order='C')
    faces = [[] for i in range(nfuns)]

    if nsubpoints == 3:
        base[:] = 1
        for i in range(nfuns):
            faces[i].append(np.arange(ndim+1)[None])
        return base.astype(bool), faces

    for tri in regions:
        for j in range(nsubpoints):
            for k in range(ndim):
                graph[j,k] = points[tri[j],k]
        graph[-1,:-1] = graph[:-1,:-1].sum(axis=0)/nsubpoints

        for i in range(nfuns):
            for j in range(nsubpoints):
                graph[j,-1] = vals[i,tri[j]]
            graph[-1,-1] = graph[:,-1].max() + 0.1
            hull = getConvexHullND(ndim, <double*>graph.data,
                                   graph.shape[0],
                                   <char*>base_ij.data)
            nphull = np.empty((hull.size(),), dtype=np.int64, order='C')
            nphull[:] = hull
            faces[i].append(nphull.reshape(-1,ndim+1))
            base[i,tri] = base_ij[:-1]

    return base.astype(bool), faces
