from setuptools import setup, find_packages

setup(
  name = 'mp-nerf',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'MP-NeRF: Massively Parallel Natural Extension of Reference Frame',
  author = 'Eric Alcaide',
  author_email = 'ericalcaide1@gmail.com',
  url = 'https://github.com/eleutherAI/mp_nerf',
  keywords = [
    'computational biolgy',
    'bioinformatics',
    'machine learning' 
  ],
  install_requires=[
    'einops>=0.3',
    'numpy',
    'torch>=1.6',
    'sidechainnet'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
