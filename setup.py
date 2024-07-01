"""
This script is used to configure the setup of the LandslideML package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='landslideml',
    version='0.0.1',
    description='A package for generation of binary landslide susceptibility ML models',
    packages=find_packages(),
    long_description=long_description,
    url='https://github.com/ImProta/LandslideML',
    author='ImProta',
    author_email="victor.improta.moreira@rwth-aachen.de",
    license='GNU',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        # add other dependencies if needed
    ],
    extras_require={
        'dev': ['pytest','black'],
    },
    python_requires='>=3.10',
)
