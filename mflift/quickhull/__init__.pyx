
import numpy as np
cimport numpy as np

from convexhull cimport getConvexHull
from libcpp.vector cimport vector

def piecewise_convexify(np.ndarray[np.float64_t, ndim=2] points,
                        np.ndarray[np.float64_t, ndim=2] vals,
                        np.ndarray[np.int64_t, ndim=2] trisubs):
    """ Piecewise convexify functions sampled on points

    Args:
        points : array of floats, shape (npoints,2)
            Set of points in the plane.
        vals : array of floats, shape (N,npoints)
        trisubs : array of ints, shape (ntris,ntrisubs)
            Indices of points that are contained in each triangle. For each
            triangle, the first three points are expected to be the triangle's
            vertices.

    Returns:
        bases : array of bools, shape (N,npoints)
            Mask indicating the base points that span the piecewise convex hull.
        faces : N lists of ntris arrays of ints, shape (nfaces,3) each
            faces[i][j] contains indices into trisubs[j].
    """
    cdef np.ndarray[np.float64_t, ndim=2] graph
    cdef np.ndarray[np.int8_t, ndim=2] bases
    cdef np.ndarray[np.int8_t, ndim=1] base
    cdef vector[size_t] hull
    cdef np.ndarray[np.int64_t, ndim=2] hulla

    ntris = trisubs.shape[0]
    ntrisubs = trisubs.shape[1]
    N = vals.shape[0]
    npoints = vals.shape[1]
    graph = np.empty((ntrisubs,3), dtype=np.float64, order='C')
    bases = np.zeros((N,npoints), dtype=np.int8, order='C')
    base = np.empty(ntrisubs, dtype=np.int8, order='C')
    faces = [[] for i in range(N)]

    if ntrisubs == 3:
        bases[:] = 1
        return bases.astype(bool)

    for tri in trisubs:
        for j in range(ntrisubs):
            graph[j,0] = points[tri[j],0]
            graph[j,1] = points[tri[j],1]

        for i in range(N):
            for j in range(ntrisubs):
                graph[j,2] = vals[i,tri[j]]
            hull = getConvexHull(<double*>graph.data,
                                 graph.shape[0],
                                 <char*>base.data)
            hulla = np.empty((hull.size()/3,3), dtype=np.int64, order='C')
            for k in range(hull.size()/3):
                hulla[k,:] = hull[3*k:3*k+3]
            faces[i].append(hulla)
            bases[i,tri] = base

    return bases.astype(bool), faces
