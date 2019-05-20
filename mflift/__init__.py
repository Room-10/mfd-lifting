
import logging
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from repyducible.experiment import Experiment as BaseExperiment

from mflift.tools.plot import plot_polys, plot_trifuns

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

    def plot(self, record=False):
        if self.pargs.plot == "no":
            return

        if record or self.pargs.plot == "hide":
            logging.info("Recording plots...")
            outputs = [(self.result, "plot-result.pdf")]
            if self.pargs.snapshots:
                outputs += [(s, "snapshot-%02d.png" % i)
                            for i,s in enumerate(self.snapshots)]

            fig = plt.gcf()

            for res,f in outputs:
                res_x =  self.model.x.vars(res['data'][0], True)

                u_proj = self.model.proj(res_x['u'])
                plot_polys(self.data.mfd.verts, self.data.mfd.simplices)
                plt.plot(u_proj[:,0], u_proj[:,1])
                c = self.data.curve(self.data.rhoGrid)
                plt.plot(c[:,0], c[:,1])

                canvas = FigureCanvasAgg(fig)
                canvas.print_figure(os.path.join(self.output_dir, f))
                plt.close(fig)
        else:
            logging.info("Plotting results interactively...")
            res_x =  self.model.x.vars(self.result['data'][0], True)

            u_proj = self.model.proj(res_x['u'])
            crv = self.data.curve(self.data.rhoGrid)

            if self.data.mfd.nembdim == 3 or hasattr(self.data.mfd, "embed"):
                verts = self.data.mfd.verts
                subgrid = self.data.S.reshape(-1,self.data.S.shape[-1])
                if hasattr(self.data.mfd, "embed"):
                    verts = self.data.mfd.embed(verts)
                    u_proj = self.data.mfd.embed(u_proj)
                    crv = self.data.mfd.embed(crv)
                    subgrid = self.data.mfd.embed(subgrid)
                from mayavi import mlab
                x,y,z = np.hsplit(verts, 3)
                mlab.triangular_mesh(x, y, z, self.data.mfd.simplices)
                mlab.plot3d(*np.hsplit(u_proj,3), color=(0,0,1), tube_radius=.01)
                mlab.plot3d(*np.hsplit(crv,3), color=(1,0,0), tube_radius=.01)
                for cu in np.stack((crv,u_proj), axis=1):
                    mlab.plot3d(*np.hsplit(cu,3), color=(0.5,0.5,0.5), tube_radius=.005)
                mlab.points3d(*np.hsplit(subgrid,3), scale_factor=.02)
                mlab.show()
            else:
                plot_polys(self.data.mfd.verts, self.data.mfd.simplices)
                plt.plot(*np.hsplit(crv,2), c="r")
                plt.plot(*np.hsplit(u_proj,2), c="b")
                plt.scatter(*np.hsplit(self.data.S.reshape(-1,2),2),
                            c='#808080', s=10.0, marker='x')
                for cu in np.stack((crv,u_proj), axis=1):
                    plt.plot(*np.hsplit(cu,2), c='#A0A0A0', linestyle="--")
                plt.axis('equal')
                plt.show()
                self.plot(record=True)
