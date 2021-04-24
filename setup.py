from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("nyx/version.py") as f:
    exec(f.read())

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name ='Nyx',
    version = __version__,
    author ='Stan Biryukov',
    author_email ='stanley.biryukov@gmail.com',
    url = 'git@github.com:stanbiryukov/Nyx.git',
    install_requires = [requirements,],
    package_data = {'nyx':['resources/*']},
    packages = find_packages(exclude=['nyx/tests']),
    license = 'MIT',
    description='Nyx: Fast and scalable RBF interpolation',
    long_description= "Nyx is a modern approach to scipy.interpolate.Rbf, enabling efficient computations with Jax and scalable operations with PyKeops",
    keywords = ['RBF', 'interpolate', 'interpolation', 'keops', 'jax', 'pykeops', 'torch', 'pytorch', 'cuda', 'spatial', 'geospatial', 'linear-algebra', 'sklearn', 'scikit-learn'],
    classifiers = [
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)