
import logging
import os
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from repyducible.experiment import Experiment as BaseExperiment

from mflift.tools.plot import plot_curves, plot_terrain_maps, plot_spd2, \
                              plot_hue_images, plot_rcom, plot_elevation

class Experiment(BaseExperiment):
    extra_source_files = ['demo.py','README.md']

    def init_params(self, *args):
        BaseExperiment.init_params(self, *args)
        self.params['plot']['subgrid'] = True
        if self.pargs.solver == "pdhg":
            self.params['solver'].update({
                'granularity': int(1e4),
                'term_maxiter': int(5e5),
                'steps': 'adaptive',
            })

    def postprocessing(self):
        mfd = self.data.mfd
        if self.data.d_image == 1 and mfd.ndim == 2 and mfd.nembdim == 3:
            L_labels = self.data.L_labels
            N_image = self.data.N_image
            res_x =  self.model.x.vars(self.result['data'][0], True)
            u_proj = self.model.proj(res_x['u'])
            curves = self.data.I + [u_proj]
            verts = self.data.T
            N_inter = 10
            N_crv = len(curves)

            crvs = np.stack(list(map(mfd.embed, curves)), axis=1)
            txtf = os.path.join(self.output_dir, "curves.txt")
            header = ["x{0} y{0} z{0}".format(i) for i in range(N_crv)]
            np.savetxt(txtf, crvs.reshape(crvs.shape[0],-1),
                       header=" ".join(header), comments="")

            crv = np.zeros((N_inter, N_crv-1, N_image, 3), dtype=np.float64)
            c1 = curves[-1]
            for k,c in enumerate(curves[:-1]):
                for i in range(N_image):
                    crv[:,k,i,:] = mfd.embed(mfd.geodesic(c[i], c1[i], N_inter))
            header = ["x{0} y{0} z{0}".format(i) for i in range((N_crv-1)*N_image)]
            txtf = os.path.join(self.output_dir, "curveorths.txt")
            np.savetxt(txtf, crv.reshape(N_inter, -1),
                       header=" ".join(header), comments="")

            edges = []
            has_edge = np.zeros((L_labels, L_labels), dtype=bool)
            for tri in self.data.P:
                for e in itertools.combinations(tri, 2):
                    e = sorted(e)
                    if not has_edge[e[0],e[1]]:
                        has_edge[e[0],e[1]] = True
                        edges.append(e)
            crv = np.zeros((N_inter, len(edges), 3), dtype=np.float64)
            for k,(i,j) in enumerate(edges):
                crv[:,k,:] = mfd.embed(mfd.geodesic(verts[i], verts[j], N_inter))
            header = ["x{0} y{0} z{0}".format(i) for i in range(crv.shape[1])]
            txtf = os.path.join(self.output_dir, "tris.txt")
            np.savetxt(txtf, crv.reshape(N_inter, -1),
                       header=" ".join(header), comments="")
        BaseExperiment.postprocessing(self)

    def plot(self, record=False):
        if self.pargs.plot == "no":
            return

        subgrid = self.data.S.reshape(-1,self.data.S.shape[-1])
        self.params['plot'].setdefault('subgrid', True)
        if not self.params['plot']['subgrid']:
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
            if self.data.name == "dti-2d":
                I = self.data.I.reshape(self.data.imagedims + (2,2))
                Iu = u_proj.reshape(self.data.imagedims + (2,2))
                plot_spd2([I,Iu], filename=f)
            if self.data.name == "rcom":
                np.savez(os.path.join(self.output_dir, "sph.npz"), sol=u_proj[0],
                    points=self.data.points, weights=self.data.weights)
                plot_rcom(u_proj, self.data, filename=f)
            elif self.data.d_image == 1:
                plot_curves(self.data.I + [u_proj], self.data.mfd,
                    subgrid=subgrid, filename=f)
            elif self.data.d_image == 2 and self.data.name[:4] == "bull":
                I = self.data.I.reshape(self.data.imagedims + (3,))
                Iu = u_proj.reshape(self.data.imagedims + (3,))
                plot_terrain_maps([I,Iu], self.data.extra, filename=f)
            elif self.data.d_image == 2 and self.data.mfd.ndim == 1:
                I = self.data.I.reshape(self.data.imagedims)
                Iu = u_proj.reshape(self.data.imagedims)
                if self.data.name == "insar_unwrap":
                    plot_elevation(Iu, I, filename=f)
                else:
                    plot_hue_images([I,Iu], filename=f)

        if not record:
            self.plot(record=True)
