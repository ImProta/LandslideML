"""
Reader module for reading in data.
"""

import pandas as pd

def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path)
