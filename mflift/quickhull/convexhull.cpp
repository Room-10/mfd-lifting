
#include "convexhull.h"
#include "src/QuickHull.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

#define DIM (3)
#define TOL (1e-9)

inline double scalar_product(double *v1, double *v2) {
    double res = 0.0;
    for (uint i = 0; i < DIM; i++) {
        res += v1[i]*v2[i];
    }
    return res;
}

inline void cross_product(double *v1, double *v2, double *res) {
    res[0] = v1[1]*v2[2] - v1[2]*v2[1];
    res[1] = v1[2]*v2[0] - v1[0]*v2[2];
    res[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

inline void elementwise_subtract(double *v1, double *v2, double *res) {
    for (uint i = 0; i < DIM; i++) {
        res[i] = v1[i] - v2[i];
    }
}

inline void scalar_multiply(double scal, double *v) {
    for (uint i = 0; i < DIM; i++) {
        v[i] *= scal;
    }
}

inline void normalize(double *n) {
    double norm = 0;
    for (uint i = 0; i < DIM; i++) {
        norm += n[i]*n[i];
    }
    scalar_multiply(1.0/std::sqrt(norm), n);
}

inline double lower_normal(double *x, double *y, double *z, double *n) {
    double a[DIM], b[DIM];
    elementwise_subtract(y, x, a);
    elementwise_subtract(z, x, b);
    cross_product(a, b, n);
    if (n[DIM-1] > 0) {
        scalar_multiply(-1.0, n);
    }
    normalize(n);
    return scalar_product(x, n) + TOL;
}

inline bool is_base_face(size_t *f, double *v, char *b, double *n, double c) {
    double fn[DIM];
    for (int i = 0; i < DIM; i++) {
        size_t l = f[i];
        if (b[l] == 2) {
            b[l] = (scalar_product(&v[l*DIM], n) > c) ? 1 : 0;
        }
        if (b[l] == 0) {
            return false;
        }
    }
    lower_normal(&v[f[0]*DIM], &v[f[1]*DIM], &v[f[2]*DIM], fn);
    return fn[DIM-1] < -TOL;
}

std::vector<size_t> getConvexHull2D(double *verts, size_t nverts, char *base)
{
    quickhull::QuickHull<double> qh;
    std::vector<size_t> hull;
    std::vector<size_t> hullf;
    size_t *hulla;
    double eq_n[DIM];
    double eq_c;
    size_t i, k, ntris;

    base[0] = base[1] = base[2] = 1;
    for (k = 3; k < nverts; k++) {
        base[k] = 2;
    }
    hull = qh.getConvexHull(verts, nverts, true, true).getIndexBuffer();
    ntris = hull.size()/3;
    hulla = hull.data();
    eq_c = lower_normal(&verts[0*DIM], &verts[1*DIM], &verts[2*DIM], eq_n);

    for (k = 0; k < ntris; k++) {
        std::sort(&hulla[k*DIM], &hulla[k*DIM] + 3);
        if (hulla[k*DIM] == 0 && hulla[k*DIM+1] == 1 && hulla[k*DIM+2] == 2) {
            continue;
        }
        if (is_base_face(&hulla[k*DIM], verts, base, eq_n, eq_c)) {
            for (i = 0; i < DIM; i++) {
                hullf.push_back(hulla[k*DIM + i]);
            }
        }
    }

    for (k = 3; k < nverts; k++) {
        base[k] = (base[k] == 1);
    }

    if (hullf.size() == 0) {
        hullf.push_back(0);
        hullf.push_back(1);
        hullf.push_back(2);
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
            slope = 1e-14*roundl(1e14*slope);
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
