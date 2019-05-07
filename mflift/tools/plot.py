
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def plot_polys(verts, polys, facecolors=(1,1,1)):
    collection = PolyCollection([verts[p,::-1] for p in polys])
    collection.set_facecolor(facecolors)
    collection.set_edgecolor((0,0,0))
    ax = plt.gca()
    ax.add_collection(collection)
    xmin, xmax = np.amin(verts[:,1]), np.amax(verts[:,1])
    ymin, ymax = np.amin(verts[:,0]), np.amax(verts[:,0])
    ax.set_xlim([xmin-1,xmax+1])
    ax.set_ylim([ymin-1,ymax+1])
