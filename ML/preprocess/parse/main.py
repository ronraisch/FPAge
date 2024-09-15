import os

import pandas as pd

from ML.preprocess.feature_creation.main import main as fc_main
from ML.preprocess.parse.metadata import main as meta_data_main
from ML.preprocess.parse.raw_data import main as raw_data_main
from ML.standarts import DATASET_FILE_NAME


def save_feather(ds, file_name):
    """
    Save a dataset to a feather file.
    Args:
        ds (pandas.DataFrame): The input dataset.
        file_name (str): The name of the file to save the dataset to.
    """
    ds.reset_index(drop=True).to_feather(file_name)


# Function to obtain the dataset by merging metadata and raw data
def get_ds(add_features=True):
    """
    Obtain the dataset by merging metadata and raw data.

    Returns:
        pandas.DataFrame: The merged dataset.
    """
    # time this function
    import time
    start = time.time()
    ds_name = DATASET_FILE_NAME + f'_add_features_{add_features}.feather'
    if os.path.exists(ds_name):
        ds = pd.read_feather(ds_name)
        print(f'Loaded dataset from {ds_name} in {time.time() - start:.2f} seconds')
        return ds
    meta_data = meta_data_main()
    raw_data = raw_data_main(list(meta_data.file_name))
    ds = pd.merge(meta_data, raw_data, left_on='file_name', right_on='file_name')
    if add_features:
        ds = fc_main(ds)
    save_feather(ds, ds_name)
    print(f'Saved dataset to {ds_name} in {time.time() - start:.2f} seconds')
    return ds
