
import numpy as np
cimport numpy as np

from convexhull cimport getConvexHull
from libcpp.vector cimport vector

def piecewise_convexify(points, vals, regions):
    """ Convexify piecewise affine functions on given subsimplices

    Args:
        points : ndarray of floats, shape (npoints, ndim)
            Set of points on the real line (ndim=1) or on the plane (ndim=2).
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
        assert points.shape[1] in [1,2]

    if points.shape[1] == 1:
        return piecewise_convexify_1d(points, vals, regions)
    else:
        return piecewise_convexify_2d(points, vals, regions)

def piecewise_convexify_1d(points, vals, regions):
    """ Convexify piecewise affine 1-d functions on given subintervals

    Args:
        points : ndarray of floats, shape (npoints, 1)
        vals : ndarray of floats, shape (nfuns, npoints)
        regions : ndarray of floats, shape (nregions, nsubpoints)

    Returns:
        base : ndarray of bools, shape (nfuns, npoints)
        faces : nfuns lists of nregions arrays of ints, shape (nfaces,2) each
    """
    npoints, ndim = points.shape
    nfuns, npoints = vals.shape
    nregions, nsubpoints = regions.shape
    assert ndim == 1

    base = np.zeros((nfuns, npoints), dtype=bool)
    faces = [[] for i in range(nfuns)]
    for i in range(nfuns):
        for j in range(nregions):
            subpoints = points[regions[j],0]
            subvals = vals[i,regions[j]]
            ordered = subpoints.argsort()
            base_ij = convexify_1d(subpoints[ordered], subvals[ordered])
            base[i,regions[j,ordered]] = base_ij
            idx = ordered[base_ij]
            faces[i].append(np.array([idx[i:i+2] for i in range(len(idx)-1)]))
    return base, faces

def convexify_1d(points, vals):
    """ Convexify piecewise affine 1-d function

    Args:
        points : ndarray of floats, shape (npoints,)
            Assumed to be in ascending order.
        vals : ndarray of floats, shape (npoints,)
            Function values corresponding to given points

    Returns:
        ndarray of bools, shape (npoints,)
    """
    npoints = points.size
    base = np.ones((npoints,), dtype=bool)
    i = 0
    while i < npoints-1:
        m = np.around((vals[i+1:] - vals[i])/(points[i+1:] - points[i]), 14)
        j = i + m.size - m[::-1].argmin()
        base[i+1:j] = False
        i = j
    return base

def piecewise_convexify_2d(np.ndarray[np.float64_t, ndim=2] points,
                           np.ndarray[np.float64_t, ndim=2] vals,
                           np.ndarray[np.int64_t, ndim=2] regions):
    """ Piecewise convexify functions sampled on 2-d points

    Args:
        points : array of floats, shape (npoints,2)
        vals : array of floats, shape (nfuns,npoints)
        regions : array of ints, shape (nregions, nsubpoints)

    Returns:
        base : array of bools, shape (nfuns,npoints)
        faces : nfuns lists of nregions arrays of ints, shape (nfaces,3) each
    """
    cdef np.ndarray[np.float64_t, ndim=2] graph
    cdef np.ndarray[np.int8_t, ndim=2] base
    cdef np.ndarray[np.int8_t, ndim=1] base_ij
    cdef vector[size_t] hull
    cdef np.ndarray[np.int64_t, ndim=2] hulla

    nregions = regions.shape[0]
    nsubpoints = regions.shape[1]
    nfuns = vals.shape[0]
    npoints = vals.shape[1]
    graph = np.empty((nsubpoints,3), dtype=np.float64, order='C')
    base = np.zeros((nfuns,npoints), dtype=np.int8, order='C')
    base_ij = np.empty(nsubpoints, dtype=np.int8, order='C')
    faces = [[] for i in range(nfuns)]

    if nsubpoints == 3:
        base[:] = 1
        for i in range(nfuns):
            faces[i].append(np.array([[0,1,2]], dtype=np.int64, order='C'))
        return base.astype(bool), faces

    for tri in regions:
        for j in range(nsubpoints):
            graph[j,0] = points[tri[j],0]
            graph[j,1] = points[tri[j],1]

        for i in range(nfuns):
            for j in range(nsubpoints):
                graph[j,2] = vals[i,tri[j]]
            hull = getConvexHull(<double*>graph.data,
                                 graph.shape[0],
                                 <char*>base_ij.data)
            hulla = np.empty((hull.size()/3,3), dtype=np.int64, order='C')
            for k in range(hull.size()/3):
                hulla[k,:] = hull[3*k:3*k+3]
            faces[i].append(hulla)
            base[i,tri] = base_ij

    return base.astype(bool), faces
