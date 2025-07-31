import pandas as pd

def load_data_step(data_path: str = 'ebay_expanded.csv') -> pd.DataFrame:
    """
    Obtaining data step

    Args:
        data_path: path to data

    Return:
        loaded data
    """
    
    return pd.read_csv(data_path, index_col='Unnamed: 0')
