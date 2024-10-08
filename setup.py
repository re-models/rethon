from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rethon",
    version="1.0.0",
    author="Claus Beisbart, Gregor Betz, Georg Brun, Sebastian Cacean, Andreas Freivogel, Richard Lohse",
    author_email="claus.beisbart@philo.unibe.ch, gregor.betz@kit.edu, georg.brun@philo.unibe.ch, "
                 "sebastian.cacean@kit.edu, andreas.freivogel@philo.unibe.ch, richard.lohse@kit.edu",
    description="A Formal Model of Reflective Equilibrium",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/re-models/rethon",
    packages=find_packages(),
    package_dir={'rethon': 'rethon'},
    package_data={'rethon': ['config/*.json', 'test_data/*.tar.gz','test_data/*.json' ]},
    classifiers=["Programming Language :: Python :: 3.8",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    python_requires='>=3.8',
    install_requires=['theodias'],
)
