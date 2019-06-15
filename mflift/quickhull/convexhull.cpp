
#include "convexhull.h"
#include "QuickHull.hpp"
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullVertexSet.h"
#include "libqhullcpp/QhullFacetList.h"
#include <vector>
#include <cmath>
#include <algorithm>

#define TOL (1e-9)

inline double scalar_product(size_t DIM, double *v1, double *v2) {
    double res = 0.0;
    for (uint i = 0; i < DIM; i++) {
        res += v1[i]*v2[i];
    }
    return res;
}

inline double det_2d(double a11, double a12, double a21, double a22) {
    return a11*a22 - a12*a21;
}

inline double det_3d(double a11, double a12, double a13,
                     double a21, double a22, double a23,
                     double a31, double a32, double a33) {
    return a11*det_2d(a22, a23, a32, a33)
         - a21*det_2d(a12, a13, a32, a33)
         + a31*det_2d(a12, a13, a22, a23);
}

inline void hodge_3d(double *v1, double *v2, double *res) {
    res[0] =  det_2d(v1[1], v1[2], v2[1], v2[2]);
    res[1] = -det_2d(v1[0], v1[2], v2[0], v2[2]);
    res[2] =  det_2d(v1[0], v1[1], v2[0], v2[1]);
}

inline void hodge_4d(double *v1, double *v2, double *v3, double *res) {
    res[0] = -det_3d(v1[1], v1[2], v1[3],
                     v2[1], v2[2], v2[3],
                     v3[1], v3[2], v3[3]);
    res[1] =  det_3d(v1[0], v1[2], v1[3],
                     v2[0], v2[2], v2[3],
                     v3[0], v3[2], v3[3]);
    res[2] = -det_3d(v1[0], v1[1], v1[3],
                     v2[0], v2[1], v2[3],
                     v3[0], v3[1], v3[3]);
    res[3] =  det_3d(v1[0], v1[1], v1[2],
                     v2[0], v2[1], v2[2],
                     v3[0], v3[1], v3[2]);
}

inline void elementwise_subtract(size_t DIM, double *v1, double *v2, double *res) {
    for (uint i = 0; i < DIM; i++) {
        res[i] = v1[i] - v2[i];
    }
}

inline void scalar_multiply(size_t DIM, double scal, double *v) {
    for (uint i = 0; i < DIM; i++) {
        v[i] *= scal;
    }
}

inline void normalize(size_t DIM, double *n) {
    double norm = 0;
    for (uint i = 0; i < DIM; i++) {
        norm += n[i]*n[i];
    }
    if (norm >= TOL) {
        scalar_multiply(DIM, 1.0/std::sqrt(norm), n);
    }
}

inline double lower_normal_3d(double *v0, double *v1, double *v2, double *n) {
    double a[3], b[3];
    elementwise_subtract(3, v1, v0, a);
    elementwise_subtract(3, v2, v0, b);
    hodge_3d(a, b, n);
    if (n[2] > 0) {
        scalar_multiply(3, -1.0, n);
    }
    normalize(3, n);
    return scalar_product(3, v0, n) + TOL;
}

inline double lower_normal_4d(double *v0, double *v1, double *v2, double *v3, double *n) {
    double a[4], b[4], c[4];
    elementwise_subtract(4, v1, v0, a);
    elementwise_subtract(4, v2, v0, b);
    elementwise_subtract(4, v3, v0, c);
    hodge_4d(a, b, c, n);
    if (n[3] > 0) {
        scalar_multiply(4, -1.0, n);
    }
    normalize(4, n);
    return scalar_product(4, v0, n) + TOL;
}

inline bool is_base_face(size_t DIM, size_t *f, double *v, char *b, double *n, double c) {
    double fn[4];
    for (uint i = 0; i < DIM; i++) {
        size_t l = f[i];
        if (b[l] == 2) {
            b[l] = (scalar_product(DIM, &v[l*DIM], n) > c) ? 1 : 0;
        }
        if (b[l] == 0) {
            return false;
        }
    }

    if (DIM == 3) {
        lower_normal_3d(&v[f[0]*DIM], &v[f[1]*DIM], &v[f[2]*DIM], fn);
    } else {
        lower_normal_4d(&v[f[0]*DIM], &v[f[1]*DIM], &v[f[2]*DIM], &v[f[3]*DIM], fn);
    }

    return fn[DIM-1] < -TOL;
}

std::vector<size_t> getConvexHullND(size_t ndim, double *verts, size_t nverts, char *base)
{
    // only ndim = 2 and ndim = 3 are supported!

    std::vector<size_t> hull;
    std::vector<size_t> hullf;
    size_t *hulla;
    double eq_n[4];
    double eq_c;
    size_t i, k, ntris;
    size_t DIM = ndim+1;

    for (k = 0; k < nverts; k++) {
        base[k] = (k < DIM) ? 1 : 2;
    }

    if (DIM == 4) {
        // slower Qhull library works for higher dimensions
        orgQhull::Qhull q("i", DIM, nverts, verts, "Qt");
        orgQhull::QhullFacetList flst = q.facetList();
        ntris = flst.size();
        hulla = (size_t*) std::malloc(DIM*ntris*sizeof(size_t));
        k = 0;
        for(orgQhull::QhullFacetList::const_iterator i = flst.begin(); i != flst.end(); ++i){
            orgQhull::QhullFacet f = *i;
            if(flst.isSelectAll() || f.isGood()){
                orgQhull::QhullVertexSet vs = f.vertices();
                for(orgQhull::QhullVertexSet::iterator vi = vs.begin(); vi != vs.end(); ++vi) {
                    hulla[k++] = (size_t) (*vi).point().id();
                }
            }
        }
        eq_c = lower_normal_4d(&verts[0*DIM], &verts[1*DIM], &verts[2*DIM], &verts[3*DIM], eq_n);
    } else {
        // for 3D, use faster QuickHull implementation
        quickhull::QuickHull<double> qh;
        hull = qh.getConvexHull(verts, nverts, true, true).getIndexBuffer();
        ntris = hull.size()/DIM;
        hulla = hull.data();
        eq_c = lower_normal_3d(&verts[0*DIM], &verts[1*DIM], &verts[2*DIM], eq_n);
    }

    for (k = 0; k < ntris; k++) {
        std::sort(&hulla[k*DIM], &hulla[k*DIM] + DIM);
        for (i = 0; i < DIM && hulla[k*DIM + i] == i; i++);
        if (i == DIM) continue;
        if (is_base_face(DIM, &hulla[k*DIM], verts, base, eq_n, eq_c)) {
            for (i = 0; i < DIM; i++) {
                hullf.push_back(hulla[k*DIM + i]);
            }
        }
    }

    if (DIM == 4) free(hulla);

    for (k = DIM; k < nverts; k++) {
        base[k] = (base[k] == 1);
    }

    if (hullf.size() == 0) {
        for (i = 0; i < DIM; i++) {
            hullf.push_back(i);
        }
    }

    return hullf;
}

void getConvexHull1D(double *graph, size_t npoints, char *base)
{
    size_t i, m, best_m;
    long double slope, best_slope;

    for (m = 0; m < npoints; m++) {
        base[m] = 1;
    }

    i = 0;
    while (i < npoints-1) {
        best_slope = INFINITY;
        best_m = i+1;
        for (m = i+1; m < npoints; m++) {
            slope = (graph[m*2 + 1] - graph[i*2 + 1])
                   /(graph[m*2 + 0] - graph[i*2 + 0]);
            slope = TOL*roundl(slope/TOL);
            if (slope <= best_slope) {
                best_m = m;
                best_slope = slope;
            }
        }
        for (m = i+1; m < best_m; m++) {
            base[m] = 0;
        }
        i = best_m;
    }
}
