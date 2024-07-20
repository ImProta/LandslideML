# LandslideML

## About

'LandslideML' is a Python package designed to generate landslide susceptibility map using machine learning models present in the scikit-learn toolbox.
It is capable of:
- Creating a Classifier Machine Learning Model amongst tested types (Random Forest, SVM or GBM)
- Training and validating such models with scikit-learn
- Predicting the label values for a given dataset and mapping to a dataframe
- Plotting the resulting structure into a 2D map

## Features 

- Read and preprocess datasets for landslide susceptibility analysis.
- Train multiple machine learning models including SVM, Logistic Regression, K-Nearest Neighbors, Decision Trees, Gradient Boosting, and Neural Networks.
- Generate and save susceptibility maps.

## Installation

### Dependencies

Ensure you have the required dependencies installed, which are listed in `requirements.txt`. You can install them using: 

```bash
pip install -r requirements.txt
```

### User installation

To install the library, you can use the 'pip' command:

```bash
pip install landslideml
```

### Folder structure

...

### Examples

Some usage examples of the library can be found in the [examples folder](https://github.com/ImProta/LandslideML/tree/e41688e91cecac40bd05ca45650c99fd23ab7b76/examples)

### License

This software is distributed under the MIT License and further information about the license can be found in the [LICENSE file](https://github.com/ImProta/LandslideML/blob/e41688e91cecac40bd05ca45650c99fd23ab7b76/LICENSE)

### Third-party libraries

The library currently uses the following third-party libraries:
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [pandas](https://pandas.pydata.org)
- [numpy](https://numpy.org)
- [matplotlib](https://matplotlib.org)
- [xarray](https://docs.xarray.dev/en/stable/index.html)


## Contributing

The project is in the early development phase. Upon completion and delivery, contributions will be welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
