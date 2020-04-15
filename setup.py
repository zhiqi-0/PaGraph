
import setuptools

setuptools.setup(
  name=             'PaGraph',
  version=          '0.1',
  author=           'Zhiqi Lin',
  author_email=     'zhiqi.0@outlook.com',
  description=      'Partition and caching for large graph datasets.',
  long_description= 'PaGraph: Scaling GNN Training on Large Graphs via Computation-aware Caching and Partitioning',
  packages=         ['PaGraph'],
  python_requires=  '>=3.6',
)