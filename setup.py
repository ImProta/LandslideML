"""
This script is used to configure the setup of the LandslideML package.
"""

import codecs
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

VERSION = '0.0.1'
DESCRIPTION = 'A package for generation of landslide susceptibility mapping with ML models'

setup(
    name='landslideml',
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    long_description=long_description,
    url='https://github.com/ImProta/LandslideML',
    author='Victor Improta',
    author_email="victor.improta.moreira@rwth-aachen.de",
    license='GNU',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'joblib==1.4.2',
        'matplotlib==3.9.1',
        'numpy==2.0.0',
        'pandas==2.2.2',
        'scikit_learn==1.5.1',
        'seaborn==0.13.2',
        'setuptools==70.2.0',
        'xarray==2024.6.0',
    ],
    extras_require={
        'dev': ['pytest','black'],
    },
    python_requires='>=3.10',
)
