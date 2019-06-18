
from distutils import log
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np

import os
import glob
import subprocess

this_dir = os.path.dirname(__file__)
quickhull_dir = os.path.join(this_dir, "mflift", "quickhull")

sources = [os.path.join(quickhull_dir, "__init__.pyx")]
sources += glob.glob(os.path.join(quickhull_dir, "quickhull_src", "*.cpp"))
includes = [
    np.get_include(),
    os.path.join(quickhull_dir, "qhull_src/src/"),
    os.path.join(quickhull_dir, "quickhull_src/"),
]

objects = [ os.path.join(quickhull_dir, "qhull_src/lib/", f) \
            for f in ["libqhullcpp.a", "libqhullstatic_r.a"]]

quickhull = Extension("mflift.quickhull.__init__",
    sources=sources,
    language="c++",
    include_dirs=includes,
    extra_objects=objects,
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"])

class MyBuildExt(build_ext):
    def run(self):
        log.info("building libqhull")
        subprocess.run(["make",
                        "bin-lib","lib/libqhullcpp.a","lib/libqhullstatic_r.a",
                        "CXX_OPTS1=-fPIC -O3 -Isrc/ $(CXX_WARNINGS)",],
                        cwd=os.path.join(quickhull_dir, "qhull_src"))
        build_ext.run(self)

setup(
    name='mflift',
    version='2019.1',
    description='Functional lifting methods for manifold-valued image processing',
    keywords='numerics scientific experiments',
    url='https://github.com/room-10/mfd-lifting',
    project_urls={ 'Source': 'https://github.com/room-10/mfd-lifting/', },
    author='Thomas Vogt',
    author_email='vogt@mic.uni-luebeck.de',
    packages=find_packages(),
    ext_modules=cythonize(quickhull),
    install_requires=[
        "cvxpy==1.0.21",
        "matplotlib==3.0.3",
        "mayavi==4.6.2",
        "numba==0.43.1",
        "numpy==1.16.3",
        "opymize @ https://github.com/room-10/Opymize/archive/master.zip#egg=opymize-0.1",
        "repyducible @ https://github.com/room-10/Repyducible/archive/master.zip#egg=repyducible-0.1",
        "scipy==1.2.1",
        "sip==4.19.8",
        "vtk==8.1.2",
    ],
    extras_require={'cuda': ["pycuda==2018.1.1"]},
    cmdclass={'build_ext': MyBuildExt},
)
