from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rethon",
    version="0.0.1a1",
    author="Claus Beisbart, Gregor Betz, Georg Brun, Sebastian Cacean, Andreas Freivogel, Richard Lohse",
    author_email="claus.beisbart@philo.unibe.ch, gregor.betz@kit.edu, georg.brun@philo.unibe.ch, "
                 "sebastian.cacean@kit.edu, andreas.freivogel@philo.unibe.ch, richard.lohse@kit.edu",
    description="A Formal Model of Reflective Equilibrium",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/debatelab/rethon",
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3.8",
                 "License :: MIT License",
                 "Operating System :: OS Independent"],
    python_requires='>=3.8',
    install_requires=['bitarray', 'py-aiger-cnf>=2.0.0', 'pypblib>=0.0.3', 'python-sat', 'numpy',
                      'numba', 'dd', 'Deprecated', 'pandas', 'tau'],
)
