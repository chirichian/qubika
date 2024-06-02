import pandas as pd

def read_file(path:str, filename:str)->pd.DataFrame:
    """Load file from source location.

    Parameters
    ----------
    path : str
        Source location
    """
    # Load dataset
    df = pd.read_csv(path+filename)

    return df

def save_file(data:pd.DataFrame, path:str, filename:str)->None:
    """Save file to destination location.

    Parameters
    ----------
    data : pd.DataFrame
        Data to be saved
    path : str
        Destination location
    """
    data.to_csv(path+filename, index=False)