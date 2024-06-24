from setuptools import setup, find_packages

setup(
    name='landslideml',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'tensorflow' 
    ],
    entry_points={
        'console_scripts': [
            'landslideml=landslideml.cli:main',
        ],
    },
)