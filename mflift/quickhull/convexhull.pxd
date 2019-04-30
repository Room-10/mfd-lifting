
from libcpp.vector cimport vector

cdef extern from "convexhull.cpp":
    pass

cdef extern from "convexhull.h":
    vector[size_t] getConvexHull(double *verts, size_t nverts, char *base)
