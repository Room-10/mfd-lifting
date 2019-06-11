
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

import os
import glob

this_dir = os.path.dirname(__file__)
quickhull_dir = os.path.join(this_dir, "mflift", "quickhull")
quickhull_sources = glob.glob(os.path.join(quickhull_dir, "quickhull_src", "*.cpp"))
qhull_sources = [
    os.path.join(quickhull_dir, "qhull_src/src/libqhullcpp", f) for f in [
        "Coordinates.cpp",
        "PointCoordinates.cpp",
        "Qhull.cpp",
        "QhullFacet.cpp",
        "QhullFacetList.cpp",
        "QhullFacetSet.cpp",
        "QhullHyperplane.cpp",
        "QhullPoint.cpp",
        "QhullPoints.cpp",
        "QhullPointSet.cpp",
        "QhullQh.cpp",
        "QhullRidge.cpp",
        "QhullSet.cpp",
        "QhullStat.cpp",
        "QhullVertex.cpp",
        "QhullVertexSet.cpp",
        #"qt-qhull.cpp",
        "RboxPoints.cpp",
        "RoadError.cpp",
        "RoadLogEvent.cpp",
        #"usermem_r-cpp.cpp",
    ]
] + [
    os.path.join(quickhull_dir, "qhull_src/src/libqhull_r", f) for f in [
        "geom2_r.c",
        "geom_r.c",
        "global_r.c",
        "io_r.c",
        "libqhull_r.c",
        "mem_r.c",
        "merge_r.c",
        "poly2_r.c",
        "poly_r.c",
        "qset_r.c",
        "random_r.c",
        "rboxlib_r.c",
        "stat_r.c",
        "usermem_r.c",
        #"userprintf_rbox_r.c",
        #"userprintf_r.c",
        "user_r.c",
    ]
]

sources = [os.path.join(quickhull_dir, "__init__.pyx")]
sources += quickhull_sources
sources += qhull_sources
includes = [
    np.get_include(),
    os.path.join(quickhull_dir, "qhull_src/src/"),
    os.path.join(quickhull_dir, "quickhull_src/"),
]

quickhull = Extension("mflift.quickhull.__init__",
    sources=sources,
    language="c++",
    include_dirs=includes,
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"])

setup(
    name='mflift',
    packages=find_packages(),
    ext_modules=cythonize(quickhull),
    install_requires=["numpy","cython"]
)
