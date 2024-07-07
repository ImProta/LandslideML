"""
Basic package for landslide susceptibility modeling using machine learning.
"""
VALID_MODELS = ['RandomForest', 'SVM', 'GBM']

from .reader import generate_model
from .model import MlModel