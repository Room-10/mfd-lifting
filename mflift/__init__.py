
import logging
import os
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from repyducible.experiment import Experiment as BaseExperiment

from mflift.tools.plot import plot_curves, plot_terrain_maps, plot_hue_images

class Experiment(BaseExperiment):
    extra_source_files = ['demo.py','README.md']

    def init_params(self, *args):
        BaseExperiment.init_params(self, *args)
        if self.pargs.solver == "pdhg":
            self.params['solver'].update({
                'granularity': int(1e4),
                'term_maxiter': int(5e5),
                'steps': 'adaptive',
            })

    def postprocessing(self):
        mfd = self.data.mfd
        if self.data.d_image == 1 and (mfd.nembdim == 3 or hasattr(mfd, "embed")):
            L_labels = self.data.L_labels
            res_x =  self.model.x.vars(self.result['data'][0], True)
            u_proj = self.model.proj(res_x['u'])
            N = 10
            t = np.linspace(0.0, 1.0, N)

            crv = self.data.curve(self.data.rhoGrid)
            crvs = np.stack((crv, u_proj), axis=1)
            txtf = os.path.join(self.output_dir, "curves.txt")
            np.savetxt(txtf, crvs.reshape(crvs.shape[0],-1),
                       header="x0 y0 z0 x1 y1 z1", comments="")
            crv = np.zeros((N, crv.shape[0], 3), dtype=np.float64)
            for k,c in enumerate(crvs):
                crv[:,k,:] = (1 - t[:,None])*c[0,None]
                crv[:,k,:] += t[:,None]*c[1,None]
                crv[:,k,:] /= np.linalg.norm(crv[:,k,:], axis=-1)[:,None]
            header = ["x{0} y{0} z{0}".format(i) for i in range(crv.shape[1])]
            txtf = os.path.join(self.output_dir, "curveorths.txt")
            np.savetxt(txtf, crv.reshape(N,-1),
                       header=" ".join(header), comments="")
            edges = []
            has_edge = np.zeros((L_labels, L_labels), dtype=bool)
            for tri in self.data.P:
                for e in itertools.combinations(tri, 2):
                    e = sorted(e)
                    if not has_edge[e[0],e[1]]:
                        has_edge[e[0],e[1]] = True
                        edges.append(e)
            crv = np.zeros((N, len(edges), 3), dtype=np.float64)
            for k,(i,j) in enumerate(edges):
                crv[:,k,:] = (1 - t[:,None])*self.data.T[i,None]
                crv[:,k,:] += t[:,None]*self.data.T[j,None]
                crv[:,k,:] /= np.linalg.norm(crv[:,k,:], axis=-1)[:,None]
            header = ["x{0} y{0} z{0}".format(i) for i in range(crv.shape[1])]
            txtf = os.path.join(self.output_dir, "tris.txt")
            np.savetxt(txtf, crv.reshape(N, -1),
                       header=" ".join(header), comments="")
        BaseExperiment.postprocessing(self)

    def plot(self, record=False):
        if self.pargs.plot == "no":
            return

        subgrid = self.data.S.reshape(-1,self.data.S.shape[-1])
        if True or self.model.name == "rof":
            subgrid = None

        outputs = []
        if record:
            logging.info("Recording plots...")
            outputs = [(self.result, "plot-result.pdf")]
            if self.pargs.snapshots:
                outputs += [(s, "snapshot-%02d.png" % i)
                            for i,s in enumerate(self.snapshots)]
        elif self.pargs.plot != "hide":
            logging.info("Plotting results interactively...")
            outputs = [(self.result, None)]

        for res,f in outputs:
            f = f if f is None else os.path.join(self.output_dir, f)
            res_x =  self.model.x.vars(res['data'][0], True)
            u_proj = self.model.proj(res_x['u'])
            if self.data.d_image == 1:
                crv = self.data.curve(self.data.rhoGrid)
                plot_curves([crv, u_proj], self.data.mfd,
                    subgrid=subgrid, filename=f)
            elif self.data.d_image == 2 and self.data.name[:4] == "bull":
                I = self.data.I.reshape(self.data.imagedims + (3,))
                Iu = u_proj.reshape(self.data.imagedims + (3,))
                plot_terrain_maps([I,Iu], self.data.extra, filename=f)
            elif self.data.d_image == 2 and self.data.mfd.ndim == 1:
                I = self.data.I.reshape(self.data.imagedims)
                Iu = u_proj.reshape(self.data.imagedims)
                plot_hue_images([I,Iu], filename=f)

        if not record:
            self.plot(record=True)
