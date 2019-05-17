
import logging
import os

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
                'steps': 'precond',
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
            plot_polys(self.data.mfd.verts, self.data.mfd.simplices)
            plt.plot(u_proj[:,0], u_proj[:,1])
            c = self.data.curve(self.data.rhoGrid)
            plt.plot(c[:,0], c[:,1])

            R = self.data.R.reshape(self.data.M_tris, self.data.N_image, -1)
            Rbase = self.data.Rbase.reshape(self.data.M_tris, self.data.N_image, -1)
            plot_trifuns(self.data.S, [[(R[:,i],Rbase[:,i]) for i in [0,1,2]],
                                       [(R[:,i],Rbase[:,i]) for i in [3,4,5]]])

            plt.show()
            self.plot(record=True)
