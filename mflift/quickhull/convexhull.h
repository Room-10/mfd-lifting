
#ifndef CONVEXHULL_H
#define CONVEXHULL_H

void getConvexHull1D(double *graph, size_t npoints, char *base);
std::vector<size_t> getConvexHull2D(double *verts, size_t nverts, char *base);

#endif
