
from libcpp.vector cimport vector

cdef extern from "convexhull.cpp":
    pass

cdef extern from "convexhull.h":
    void getConvexHull1D(double *graph, size_t npoints, char *base)
    vector[size_t] getConvexHull2D(double *verts, size_t nverts, char *base)
