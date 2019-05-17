
import numpy as np

try:
    import mflift
except:
    import set_path
from mflift.tools.linalg import gramschmidt

def test_gramschmidt(A):
    nbases, nvecs, ndim = A.shape
    Ags = gramschmidt(A)
    err = np.eye(nvecs)[None] - np.einsum('ikm,ilm->ikl', Ags, Ags)
    assert np.linalg.norm(err.ravel()) < 1e-10

def main():
    print("=> Testing Gram-Schmidt orthonormalization")
    test_gramschmidt(np.array([
        [[1,0,0],[1,0,1],[1,1,0]],
        [[0,3,0],[2,0,0],[0,0,-2]],
        [[1,1,0],[2,0,1],[1,1,2]],
        [[1,1,-1],[-2,1,1],[1,-2,1]],
    ], dtype=np.float64))
    test_gramschmidt(np.array([
        [[1,0,0,-1],[1,0,1,2]],
        [[0,3,0,1],[2,0,0,0]],
    ], dtype=np.float64))

if __name__ == "__main__":
    main()