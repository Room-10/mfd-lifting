
import numpy as np

try:
    import precise
except:
    import set_path
from mflift.quickhull import piecewise_convexify

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
args = (points, vals, trisubs)

for i in range(10):
    # shift, rotate and stretch domain
    phi = np.random.rand()
    r = 0.2 + 2*np.random.rand(2)
    v = -6 + 12*np.random.rand(1,2)
    R = np.array([[r[0]*np.cos(phi), -r[1]*np.sin(phi)],
                  [r[0]*np.sin(phi),  r[1]*np.cos(phi)]])
    points[:] = points.dot(R.T) + v

    # add affine function to values
    b = -5 + 10*np.random.rand(2,1)
    c = -4 + 8*np.random.rand()
    vals += points.dot(b).T + c

    assert np.all(piecewise_convexify(*args)[0] == result)

vals[:] = 0
result[0,3:] = False
assert np.all(piecewise_convexify(*args)[0] == result)


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
args = (points, vals, trisubs)

for i in range(10):
    # shift, rotate and stretch domain
    r = 0.2 + 2*np.random.rand()
    v = -6 + 12*np.random.rand()
    points[:] = r*points + v

    # add affine function to values
    b = -5 + 10*np.random.rand()
    c = -4 + 8*np.random.rand()
    vals += b*points.T + c

    assert np.all(piecewise_convexify(*args)[0] == result)
