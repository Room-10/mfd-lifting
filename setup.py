
from setuptools import setup, find_packages

opymize_base_url = "https://github.com/Room-10/Opymize"
opymize_version = "0.1"
opymize_url = "{url}/releases/download/v{ver}/opymize-{ver}.tar.gz".format(
              url=opymize_base_url, ver=opymize_version)
opymize_pkg = "opymize @ {0}#egg=opymize-{1}".format(opymize_url,opymize_version)

repyd_base_url = "https://github.com/room-10/Repyducible"
repyd_version = "0.1"
repyd_pkg = "repyducible @ {url}/archive/master.zip#egg=opymize-{ver}".format(
            url=repyd_base_url, ver=repyd_version)

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
    install_requires=[
        "cvxpy==1.0.21",
        "matplotlib==3.0.3",
        "mayavi==4.6.2",
        "numba==0.43.1",
        "numpy==1.16.3",
        opymize_pkg,
        repyd_pkg,
        "scikit-image==0.15.0",
        "scipy==1.10.0",
        "sip==4.19.8",
        "vtk==8.1.2",
    ],
    extras_require={'cuda': ["pycuda==2018.1.1"]},
)
