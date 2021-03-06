
import itertools
import numpy as np
from scipy.io import loadmat

from mflift.manifolds import DiscretizedManifold
from mflift.tools.linalg import normalize, quaternion_so3

class SO3(DiscretizedManifold):
    """ 3-dimensional rotational group represented using unit quaternions """
    ndim = 3
    nembdim = 9

    def __init__(self, h):
        """ Setup a simplicial grid on SO(3).

        Args:
            h : maximal length of edges in the triangulation
        """
        self.verts, self.simplices = so3mesh_hexacosichoron()
        DiscretizedManifold.__init__(self, h)

    def mesh(self, h):
        triverts = self.verts[self.simplices]
        maxedgelen = self.dist(triverts, triverts).max()
        rep = max(0, np.ceil(np.log2(maxedgelen/h)))
        return self.mesh_refine(self.verts, self.simplices, repeat=rep)

    def mesh_refine(self, verts, simplices, repeat=1):
        """ Refine a triangulation of SO(3)

        The algorithm applies the following procedure to a given triangulation:
        Each tetrahedron is split into eight tetrahedra using the tetrahedron's edge
        centers and its barycenter as new vertices.

        Args:
            verts : ndarray of floats, shape (nverts,4)
            simplices : ndarray of floats, shape (ntris,4)
            repeat : int
                The refinement procedure is iterated `repeat` times.
                If `repeat` is 0, the input is returned unchanged.

        Returns:
            verts : ndarray of floats, shape (nverts,4)
            simplices : ndarray of floats, shape (ntris,4)
        """
        if repeat == 0: return verts, simplices

        nverts = verts.shape[0]
        nsimplices = simplices.shape[0]

        edges = []
        edgecenters = np.zeros((nverts, nverts), dtype=np.int64)
        for j,sim in enumerate(simplices):
            for e in itertools.combinations(sim, 2):
                if edgecenters[e[0],e[1]] == 0:
                    edges.append(e)
                    edgecenters[e[0],e[1]] = edgecenters[e[1],e[0]] = nverts
                    nverts += 1
        edges = np.array(edges, dtype=np.int64)
        barycenters = np.int64(np.arange(nverts, nverts + nsimplices))
        nverts += nsimplices
        verts = np.concatenate((verts,
            self.mean(verts[edges][None], np.ones((1,1,2)))[0,:,0],
            self.mean(verts[simplices][None], np.ones((1,1,4)))[0,:,0],), axis=0)

        newsims = []
        for j,sim in enumerate(simplices):
            p12 = edgecenters[sim[0],sim[1]]
            p23 = edgecenters[sim[1],sim[2]]
            p13 = edgecenters[sim[0],sim[2]]
            p14 = edgecenters[sim[0],sim[3]]
            p24 = edgecenters[sim[1],sim[3]]
            p34 = edgecenters[sim[2],sim[3]]
            pc = barycenters[j]
            assert np.all([p12, p23, p13, p14, p24, p34, pc])
            newsims.extend([
                [sim[0], p12, p13, p14],
                [    pc, p12, p13, p14],
                [sim[1], p12, p23, p24],
                [    pc, p12, p23, p24],
                [sim[2], p13, p23, p34],
                [    pc, p13, p23, p34],
                [sim[3], p14, p24, p34],
                [    pc, p14, p24, p34],
            ])
        simplices = np.asarray(newsims)

        return self.mesh_refine(verts, simplices, repeat=repeat-1)

    def _log(self, location, pfrom, out):
        """ exp_l^{-1}(p) = d(l,p)*(sign(<p,l>)p - |<p,l>|l)/|p - <p,l>l| """
        # pl : <p,l>
        # fc : d(l,p)/|p - <p,l>*l| = arccos(|pl|)/sqrt(1 - pl^2)
        # out : fc*(sign(pl)*p - |pl|*l)
        pl = np.clip(np.einsum('ilm,ikm->ikl', pfrom, location), -1.0, 1.0)
        sign_pl = np.sign(pl[:,:,:,None])
        sign_pl[sign_pl == 0] = 1
        pl = np.abs(pl[:,:,:,None])
        fc = np.arccos(pl)/np.fmax(np.spacing(1),np.sqrt(1 - pl**2))
        out[:] = fc*(sign_pl*pfrom[:,None,:,:] - pl*location[:,:,None,:])

    def _exp(self, location, vfrom, out):
        """ exp_l(v) = cos(|v|) * l  + sin(|v|) * v/|v| """
        vn = np.sqrt(np.einsum('ikm,ikm->ik', vfrom, vfrom))
        vnm = np.fmax(np.spacing(1),vn[:,None,:,None])
        out[:] = np.cos(vn[:,None,:,None])*location[:,:,None,:]
        out += np.sin(vn[:,None,:,None])/vnm*vfrom[:,None,:,:]
        # normalizing prevents errors from accumulating
        normalize(out)

    def _dist(self, x, y, out):
        np.einsum('ikm,ilm->ikl', x, y, out=out)
        out[:]= np.arccos(np.abs(np.clip(out, -1.0, 1.0)))

    def embed(self, x):
        return quaternion_so3(x).reshape(x.shape[:-1] + (9,))

def so3mesh_hexacosichoron():
    """ 4-d regular hexacosichoron (600-cell) where opposite points are
        identified (one of them is removed)

    Returns:
        verts : ndarray of floats, shape (60,4)
            Each row corresponds to a point on the 3-sphere.
        simplices : ndarray of floats, shape (300,4)
            Each row defines a simplex through indices into `verts`.
    """
    data = loadmat("data/hcc.mat")["hcc"]
    verts, simplices = data["vertices"][0,0].T, data["faces"][0,0].T-1
    normals = data["normals"][0,0].T
    opverts = data["opvertices"][0,0][:,0]-1

    # throw out half of all simplices based on orientation of its normal
    # vector with respect to the fixed reference direction
    reference_dir = np.array([1.0, 1e-4, 1.1e-4, 1.5e-4])
    simplices = simplices[normals.dot(reference_dir) > 0,:]
    newinds = np.zeros((verts.shape[0],), dtype=np.int64)
    vertkeep = (verts.dot(reference_dir) > 0)
    vertdiscard = np.logical_not(vertkeep)
    newinds[vertkeep] = np.arange(60)
    newinds[vertdiscard] = newinds[opverts[vertdiscard]]
    return verts[vertkeep], np.ascontiguousarray(newinds[simplices])
