from setuptools import setup

setup(name='netgan',
      version='0.1',
      description='NetGAN: Generating Graphs via Random Walks',
      author='Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, Stephan Günnemann',
      author_email='zuegnerd@in.tum.de',
      packages=['netgan'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'tensorflow', 'networkx', 'numba'],
zip_safe=False)