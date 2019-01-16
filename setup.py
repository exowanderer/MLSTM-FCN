try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='mlstmfcn',
      version=0.1,
      # cmdclass=versioneer.get_cmdclass(),
      description='Multivariate LSTM Fully Convolutional Networks for Time Series Classification',
      long_description=open('README.md').read(),
      url='https://github.com/exowanderer/mlstm_fcn',
      license='GPL3',
      author="Somshubra Majumdar(titu1994), fazlekarim, "\
              "Jonathan Fraine (exowanderer)",
      packages=find_packages(),
      install_requires=['tensorflow>=1.4.0', 'keras>=2.1.2', 'scipy', 
                        'numpy', 'numpy', 'pandas',
                        'scikit-learn>=0.18.2', 'h5py'],
      extras_require={'plots':  ["matplotlib"]}
      )