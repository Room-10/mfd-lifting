
import logging
import numpy as np

from opymize import Variable
from opymize.functionals import SplitSum, ZeroFct, IndicatorFct, PositivityFct, \
                                QuadEpiSupp, EpigraphSupp, L1Norms, HuberPerspective
from opymize.linear import BlockOp, IdentityOp, GradientOp, \
                           IndexedMultAdj, MatrixMultR, MatrixMultRBatched

from mflift.models import SublabelModel

class Model(SublabelModel):
    name = "rof"

    def __init__(self, *args, lbd=1.0, regularizer="tv", alph=np.inf,
                              fdscheme="centered", **kwargs):
        SublabelModel.__init__(self, *args, **kwargs)
        self.lbd = lbd
        self.regularizer = regularizer
        self.alph = alph
        self.fdscheme = fdscheme
        logging.info("Init model '%s' (%s regularizer, lambda=%.2e, "
                                      "alpha=%.2e, fdscheme=%s)" \
                     % (self.name, self.regularizer, self.lbd,
                        self.alph, self.fdscheme))

        imagedims = self.data.imagedims
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        M_tris = self.data.M_tris
        s_gamma = self.data.s_gamma
        d_image = self.data.d_image

        xvars = [('u', (N_image, L_labels)),
                 ('w12', (M_tris, N_image, s_gamma+1)),
                 ('w', (M_tris, N_image, d_image, s_gamma))]
        yvars = [('p', (N_image, d_image, L_labels)),
                 ('q', (N_image, L_labels)),
                 ('v12a', (M_tris, N_image, s_gamma+1)),
                 ('v12b', (M_tris, N_image, s_gamma+1)),
                 ('v3', (N_image,)),]

        if self.regularizer == "tv":
            yvars.append(('g', (M_tris, N_image, d_image, s_gamma)))
        elif self.regularizer == "quadratic":
            yvars.append(('g12', (M_tris, N_image, d_image*s_gamma+1)))

        self.x = Variable(*xvars)
        self.y = Variable(*yvars)

    def setup_solver(self, *args):
        imagedims = self.data.imagedims
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        M_tris = self.data.M_tris
        s_gamma = self.data.s_gamma
        d_image = self.data.d_image

        Id_w2 = np.zeros((s_gamma+1,d_image*s_gamma+1), order='C')
        Id_w2[-1,-1] = 1.0

        Adext = np.zeros((M_tris,s_gamma,d_image*s_gamma+1), order='C')
        Adext[:,:,:-1] = np.tile(self.data.Ad, (1,1,d_image))

        # Ab (M_tris, s_gamma+1, s_gamma+1)
        Ab_mats = -np.ones((M_tris, s_gamma+1, s_gamma+1),
                            dtype=np.float64, order='C')
        Ab_mats[:,:,0:-1] = self.data.T[self.data.P]
        Ab_mats[:] = np.linalg.inv(Ab_mats)

        self.linblocks.update({
            'PAbTri': IndexedMultAdj(L_labels, N_image, self.data.P, Ab_mats),
            'Grad': GradientOp(imagedims, L_labels, scheme=self.fdscheme),
            'PB': IndexedMultAdj(L_labels, d_image*N_image, self.data.P, self.data.B),
            'Ad': MatrixMultRBatched(N_image*d_image, self.data.Ad),
            'Adext': MatrixMultRBatched(N_image, Adext),
            'Id_w2': MatrixMultR(M_tris*N_image, Id_w2),
        })
        SublabelModel.setup_solver(self, *args)

    def setup_solver_pdhg(self):
        x, y = self.x.vars(named=True), self.y.vars(named=True)
        imagedims = self.data.imagedims
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        M_tris = self.data.M_tris
        s_gamma = self.data.s_gamma
        d_image = self.data.d_image

        PAbOp = self.linblocks['PAbTri']
        S_u_k = self.linblocks['S']
        GradOp = self.linblocks['Grad']
        PBLinOp = self.linblocks['PB']
        AdMult = self.linblocks['Ad']

        shift = np.tile(self.data.data_b, (M_tris,1,1)).reshape((-1, s_gamma))
        c = 0.5*(shift**2).sum(axis=-1)
        epifct1 = QuadEpiSupp(M_tris*N_image, s_gamma, b=-shift, c=c)
        epifct2 = EpigraphSupp(np.ones((N_image, L_labels), dtype=bool),
            [[np.arange(s_gamma+1)[None]]*M_tris]*N_image,
            self.data.P, self.data.T, np.zeros((N_image, L_labels)))

        Id_u = IdentityOp(x['u']['size'])
        Id_w12 = IdentityOp(x['w12']['size'])

        if self.data.constraints is not None:
            constrmask, constru = self.data.constraints
            constru_lifted = self.data.mfd.embed_barycentric(constru)[1]
            Gu = ConstrainFct(constrmask, constru_lifted)
        else:
            Gu = PositivityFct(x['u']['size'])

        self.pdhg_G = SplitSum([
            Gu,                         # \delta_{u >= 0} or constraints
            ZeroFct(x['w12']['size']),  # 0
            ZeroFct(x['w']['size']),    # 0
        ])

        F_summands = [
            IndicatorFct(y['p']['size']),        # \delta_{p = 0}
            IndicatorFct(y['q']['size']),        # \delta_{q = 0}
            epifct1,                             # -0.5*v2a*|v1a/v2a + b|^2
            epifct2,                             # \max_{v \in Delta} <v12b,v>
            IndicatorFct(y['v3']['size'], c1=1), # \delta_{v3^i = 1}
        ]

        op_blocks = [
            [GradOp,       0, PBLinOp], # p = Du - P'B'w
            [  Id_u,   PAbOp,       0], # q = u - P'Ab'w12
            [     0,  Id_w12,       0], # v12a = w12
            [     0,  Id_w12,       0], # v12b = w12
            [ S_u_k,       0,       0], # v3^i = sum_k u[i,k]
        ]

        if self.regularizer == "tv":
            l1norms = L1Norms(M_tris*N_image, (d_image, s_gamma), self.lbd, "nuclear")
            F_summands.append(l1norms) # lbd*\sum_ji |g[j,i,:,:]|_nuc
            op_blocks.append([     0,       0,  AdMult]) # g = A'w
        elif self.regularizer == "quadratic":
            if self.alph < np.inf:
                etahat = HuberPerspective(M_tris*N_image, s_gamma*d_image,
                                          lbd=self.lbd, alph=self.alph)
            else:
                etahat = QuadEpiSupp(M_tris*N_image, s_gamma*d_image, a=self.lbd)
            F_summands.append(etahat) # 0.5*lbd*\sum_ji |g1[j,i]|^2/|g2[j,i]|
            AdMult = self.linblocks['Adext']
            Id_w2 = self.linblocks['Id_w2']
            op_blocks.append([     0,   Id_w2,   AdMult]) # g12 = (Ad'w, w2)
        self.pdhg_F = SplitSum(F_summands)
        self.pdhg_linop = BlockOp(op_blocks)
