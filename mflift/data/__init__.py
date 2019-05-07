
import logging
import numpy as np

from repyducible.data import Data as BaseData

from mflift.quickhull import piecewise_convexify

class ManifoldValuedData(BaseData):
    rho = None
    N = 0
    initializer = None
    constraints = None

    def __init__(self, mfd, *args, **kwargs):
        BaseData.__init__(self, *args, **kwargs)

        self.mfd = mfd
        self.s_gamma = self.mfd.ndim

        # T (L_labels, s_gamma)
        # P (M_tris, s_gamma+1)
        self.T, self.P = self.mfd.verts, self.mfd.simplices
        self.L_labels = self.mfd.nverts
        self.M_tris = self.mfd.nsimplices

        # Ab (M_tris, s_gamma+1, s_gamma+1)
        self.Ab_mats = np.zeros((self.M_tris, self.s_gamma+1, self.s_gamma+1),
                                dtype=np.float64, order='C')
        self.Ab_mats[:] = self.mfd.barycentric_coordinate_matrices.transpose(0,2,1)

        # B (M_tris, s_gamma, s_gamma+1)
        # Ad (M_tris, s_gamma, s_gamma)
        self.B, self.Ad = self.mfd.derivative_matrices

        # S (l_sublabels, s_gamma)
        # Q (M_tris, m_sublabels)
        self.S, self.Q =  self.mfd.subgrid(self.l_dimsubls)
        self.l_sublabels = self.S.shape[0]
        self.m_sublabels = self.Q.shape[1]

        logging.info("Label space: %d labels, %d tris, %d sublabels (%d per tri)"
            % (self.L_labels, self.M_tris, self.l_sublabels, self.m_sublabels))

        # R (N*M_tris, l_sublabels)
        # Rbase (N*M_tris, l_sublabels)
        npoints = self.N*self.l_sublabels
        nhulls = self.N*self.M_tris
        logging.info("Evaluating function on %d points..." % npoints)
        self.R = self.rho_bary(self.S).reshape(-1, self.m_sublabels)
        logging.info("Computing %d convex hulls..." % nhulls)
        self.Rbase, self.Rfaces = piecewise_convexify(self.S, self.R, self.Q)

    def rho_bary(self, coords):
        M, N, m = self.M_tris, self.N_image, self.m_sublabels
        weights = np.concatenate((coords, 1 - coords.sum(axis=1)[:,None]), axis=1)
        sublabels = self.mfd.mean(self.T[self.P], weights=weights)
        result = self.rho(sublabels.reshape(-1, sublabels.shape[-1])[None,:,:])
        R = np.zeros((M,N,m), dtype=np.float64, order='C')
        R[:] = result.reshape((N,M,m)).transpose(1,0,2)
        return R

    def rho(self, z):
        return self.rho_x(self.rhoGrid, z)
