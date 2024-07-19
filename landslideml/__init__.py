"""
Basic package for landslide susceptibility modeling using machine learning.
"""

from .reader import generate_model, load_model
from .model import MlModel
from .output import plot_heatmap, compare_metrics, plot_map
