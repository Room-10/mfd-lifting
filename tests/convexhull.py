
import numpy as np

try:
    import precise
except:
    import set_path
from mflift.quickhull import piecewise_convexify
from mflift.tools.linalg import barygrid

def random_rotation(ndim):
    if ndim == 1:
        return np.ones((1,1))
    elif ndim == 2:
        phi = np.random.rand()
        return np.array([[np.cos(phi), -np.sin(phi)],
                         [np.sin(phi),  np.cos(phi)]])
    elif ndim == 3:
        v = np.random.rand(ndim)
        v /= np.linalg.norm(v)
        phi = np.random.rand()
        K = np.array([[    0, v[2],-v[1]],
                      [-v[2],    0, v[0]],
                      [ v[1],-v[0],    0]])
        return np.eye(ndim) + np.sin(phi)*K + (1 - np.cos(phi))*K.dot(K)

def test_convexify(points, vals, trisubs, result):
    ndim = points.shape[1]
    nfuns = vals.shape[0]
    epoints = points.copy()
    evals = vals.copy()
    args = (epoints, evals, trisubs)
    assert np.all(piecewise_convexify(*args)[0] == result)

    for i in range(100):
        # shift, rotate and stretch domain
        v = -6 + 12*np.random.rand(1,ndim)
        R = (0.2 + 2*np.random.rand(1,ndim))*random_rotation(ndim)
        epoints[:] = points.dot(R.T) + v

        # add affine function to values
        b = -5 + 10*np.random.rand(ndim,nfuns)
        c = -4 + 8*np.random.rand(nfuns,1)
        evals += points.dot(b).T + c

        #print(points)
        #print(vals)
        #print(trisubs)
        #print(result)
        #print(piecewise_convexify(*args)[0])
        assert np.all(piecewise_convexify(*args)[0] == result)

def test_convex_fun(f, ndim):
    bary = barygrid(ndim, 5, boundary=True)
    bary = bary[np.all(bary != 1, axis=1)]
    bary = np.concatenate((np.eye(ndim+1), bary), axis=0)
    points = np.array(bary[:,:-1], dtype=np.float64, order='C')
    npoints = points.shape[0]
    trisubs = np.arange(bary.shape[0])[None,:]
    trisubs = np.array(trisubs, dtype=np.int64, order='C')
    vals = np.zeros((1, npoints))
    vals[:] = [[f(pt) for pt in points]]
    result = np.ones((1, npoints), dtype=bool)
    if np.all(vals == 0.0):
        result[0,(ndim+1):] = False
    test_convexify(points, vals, trisubs, result)

def test_handcrafted_1d():
    points = np.array([[0.0],
                       [1.0],
                       [0.1],
                       [0.2], # 3 : dummy
                       [0.4],
                       [0.7],])
    npoints = points.shape[0]
    vals = np.zeros((1, npoints))
    vals[0,:] = [0.0, 0.0, 1.0, -0.3, -0.3, 2.0]
    trisubs = np.array([0,1,5,2,4], dtype=np.int64)[None,:]
    result = np.zeros((1, npoints), dtype=bool)
    result[0,:] = [True, True, False, False, True, False]
    test_convexify(points, vals, trisubs, result)

def test_handcrafted_2d():
    points = np.array([[0.0,0.0],
                       [0.0,1.0],
                       [1.0,1.0],
                       [0.5,0.5],
                       [0.0,0.2], # 4 : dummy
                       [0.0,0.5],
                       [0.5,1.0],
                       [0.3,0.7]])
    npoints = points.shape[0]
    vals = np.zeros((1, npoints))
    vals[0,:] = [0.0, 0.0, 0.0, 1.0, 2.0, 0.3, -0.3, -0.3]
    trisubs = np.array([0,1,2,5,6,7,3], dtype=np.int64)[None,:]
    result = np.zeros((1, npoints), dtype=bool)
    result[0,:] = [True, True, True, False, False, False, True, True]
    test_convexify(points, vals, trisubs, result)

def main():
    test_convex_fun(lambda x: 0.0, 1)
    test_convex_fun(lambda x: x[0]**2, 1)
    test_handcrafted_1d()
    test_convex_fun(lambda x: 0.0, 2)
    test_convex_fun(lambda x: (x**2).sum(), 2)
    test_handcrafted_2d()
    test_convex_fun(lambda x: 0.0, 3)
    test_convex_fun(lambda x: (x**2).sum(), 3)

if __name__ == "__main__":
    main()
