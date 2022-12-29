from distutils.core import setup

setup(name='genericSolver',
      version='1.0',
      description='SEP Python solver',
      author='E.Biondi, ...., Clapp',
      author_email='bob@sep.stanford.edu',
      url="http://zapad.stanford.edu/bob/python-solver/-/tree/main/GenericSolver",
          packages=['genericSolver'],
    install_requires=[ 'numpy>=1.18.1', 'pylops>=2.0.0', 
                       'h5py>=2.10.0', 'distributed>=2021.1.1','dask-jobqueue>=0.5.5',
                      'matplotlib>=3.3.4', 'scipy>=1.4.2']
   )
