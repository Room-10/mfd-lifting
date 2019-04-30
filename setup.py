
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

import os
import glob

this_dir = os.path.dirname(__file__)
quickhull_dir = os.path.join(this_dir, "mflift", "quickhull")
quickhull_sources = glob.glob(os.path.join(quickhull_dir, "src", "*.cpp"))
sources = [os.path.join(quickhull_dir, "__init__.pyx")] + quickhull_sources

quickhull = Extension("mflift.quickhull.__init__",
    sources=sources,
    language="c++",
    include_dirs=[np.get_include()],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"])

setup(
    name='mflift',
    packages=find_packages(),
    ext_modules=cythonize(quickhull),
    install_requires=["numpy","cython"]
)
