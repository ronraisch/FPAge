# Import necessary libraries
import warnings

import pandas as pd
from numpy.random import RandomState
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ML.preprocess.parse.main import get_ds
from ML.standarts import AMOUNT_COL_NAME, RANDOM_STATE, TEST_SIZE, TARGET_LABEL, VAL_SIZE, \
    NORM_COL_NAME, \
    DENOISE_COL_NAME, SUM_COL_NAME, PCA_COL_NAME


# Function to retrieve column names starting with a given prefix from a dataset
def get_columns_prefix(ds, col_prefix=AMOUNT_COL_NAME):
    """
    Retrieve column names starting with a given prefix from a dataset.

    Args:
        ds (pandas.DataFrame): The input dataset.
        col_prefix (str): Prefix to filter column names. Default is 'AMOUNT_COL_NAME'.

    Returns:
        list: List of column names that match the prefix.
    """
    return [col for col in ds.columns if col.startswith(col_prefix)]


def get_x_columns(ds, prefix_list):
    cols = []
    for prefix in prefix_list:
        cols += get_columns_prefix(ds, prefix)
    return cols


# Function to retrieve the target column name
def get_y_columns():
    """
    Get the target column name.

    Returns:
        str: The name of the target column.
    """
    return TARGET_LABEL


# Function to unpack features (X) and target (y) from a dataset
def unpack_ds(ds):
    """
    Unpack features (X) and target (y) from a dataset.

    Args:
        ds (pandas.DataFrame): The input dataset.

    Returns:
        tuple: A tuple containing two pandas.DataFrames - X (features) and y (target).
    """
    X = ds[get_x_columns(ds, prefix_list=[AMOUNT_COL_NAME, NORM_COL_NAME, DENOISE_COL_NAME,
                                          SUM_COL_NAME])]
    y = ds[get_y_columns()]
    return X, y


# Function to unpack features (X) and target (y) from a list of datasets
def unpack_ds_list(ds_list):
    """
    Unpack features (X) and target (y) from a list of datasets.

    Args:
        ds_list (list of pandas.DataFrame): List of input datasets.

    Returns:
        list: List containing concatenated features (X) and targets (y).
    """
    X_list = []
    y_list = []
    for ds in ds_list:
        X, y = unpack_ds(ds)
        X_list.append(X)
        y_list.append(y)
    return X_list + y_list


def shuffle_ds(ds, shuffle, random_state):
    if shuffle:
        return ds.sample(frac=1, random_state=random_state)
    return ds


# Function to split a dataset into selected and not selected portions based on slide index
def split_ds_by_slide(ds, split_size, random_state):
    """
    Split a dataset into selected and not selected portions based on slide index.

    Args:
        ds (pandas.DataFrame): The input dataset.
        split_size (float): The size of the selected portion (test size).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Two pandas.DataFrames - ds_selected (selected portion) and ds_not_selected (not selected portion).
    """
    slide_indices = list(ds.slide_idx.unique())
    split_length = int(len(slide_indices) * (1 - split_size))
    rng = RandomState(random_state)
    selected_slides = rng.choice(slide_indices, split_length, replace=False)
    ds_selected = ds[ds.slide_idx.isin(selected_slides)]
    ds_not_selected = ds[~ds.slide_idx.isin(selected_slides)]
    return ds_selected, ds_not_selected


# Function to split a dataset into multiple selected portions based on slide index
def slide_multiple_split(ds, split_size, random_state, n_splits=None):
    """
    Split a dataset into multiple selected portions based on slide index.

    Args:
        ds (pandas.DataFrame): The input dataset.
        split_size (float or list of float): The size of the selected portions (test size) as a list.
        random_state (int): Random seed for reproducibility.
        n_splits (int): Number of splits. If None, it's determined by the length of split_size.

    Returns:
        list: List of pandas.DataFrames - selected portions.
    """
    ds_list = []
    assert type(split_size) is list or n_splits is not None, ('split_size must be given as list or '
                                                              'n_splits must be specified')
    if type(split_size) is float:
        split_size = [split_size] * n_splits
    if n_splits is None:
        n_splits = len(split_size)
    for i in range(n_splits):
        ds, ds_split = split_ds_by_slide(ds, split_size=split_size[i] / (1 - sum(split_size[:i])),
                                         random_state=random_state)
        ds_list.append(ds_split)
    ds_list = [ds] + ds_list
    return ds_list


# Function to split a dataset into training and test sets based on slide index
def slide_ds_train_test_split(ds, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True) -> list[pd.DataFrame]:
    """
    Split a dataset into training and test sets based on slide index.

    Args:
        ds (pandas.DataFrame): The input dataset.
        test_size (float): The size of the test set.
        random_state (int): Random seed for reproducibility.
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        tuple: Two pandas.DataFrames - ds_train (training set) and ds_test (test set).
    """
    ds = shuffle_ds(ds, shuffle, random_state)
    ds_train, ds_test = split_ds_by_slide(ds, split_size=test_size, random_state=random_state)
    return [ds_train, ds_test]


# Function to split a dataset into training, validation, and test sets based on slide index
def slide_ds_train_val_test_split(ds, test_size=TEST_SIZE, val_size=VAL_SIZE, shuffle=True,
                                  random_state=RANDOM_STATE) -> list[pd.DataFrame]:
    """
    Split a dataset into training, validation, and test sets based on slide index.

    Args:
        ds (pandas.DataFrame): The input dataset.
        test_size (float): The size of the test set.
        val_size (float): The size of the validation set.
        shuffle (bool): Whether to shuffle the dataset before splitting.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Three pandas.DataFrames - ds_train (training set), ds_val (validation set), and ds_test (test set).
    """
    ds = shuffle_ds(ds, shuffle, random_state)
    ds_train, ds_val, ds_test = slide_multiple_split(ds, split_size=[val_size, test_size],
                                                     random_state=random_state)
    return [ds_train, ds_val, ds_test]


def apply_data_transformation_split(unpack_list, transformer):
    """
    Apply a data transformation to a list of datasets.
    :param unpack_list:
    :param transformer:
    :return:
    """
    transformer.fit(unpack_list[0])
    for i in range(len(unpack_list) // 2):
        unpack_list[i] = pd.DataFrame(transformer.transform(unpack_list[i]), index=unpack_list[
            i].index)
    return unpack_list


def pca_split(unpack_list):
    X_train_shape = unpack_list[0].shape
    pca = PCA(n_components=min(X_train_shape[0], X_train_shape[1]) - 1)
    scaled_unpack_list = standard_scalar_split(unpack_list)
    pca_unpack_list = apply_data_transformation_split(scaled_unpack_list, pca)
    for i in range(len(pca_unpack_list) // 2):
        pca_unpack_list[i] = pd.DataFrame(pca_unpack_list[i].values, index=unpack_list[i].index,
                                          columns=[f'{PCA_COL_NAME}_{i}' for i in
                                                   range(pca_unpack_list[i].shape[1])])
    return pca_unpack_list


def standard_scalar_split(unpack_list):
    scalar = StandardScaler()
    return apply_data_transformation_split(unpack_list, scalar)


# Main function to load the dataset and split it into training, validation, and test sets
def main(unpack=True, include_val=True, use_pca=True, add_features=True):
    """
    Main function to load the dataset and split it into training, validation, and test sets.

    Args:
        unpack (bool): Whether to unpack the dataset into features (X) and target (y).
        include_val (bool): Whether to include the validation set.
        use_pca (bool): Whether to use PCA for dimensionality reduction.
        add_features (bool): Whether to add features to the dataset.
    Returns:
        tuple or pandas.DataFrame: If 'unpack' is True, returns X_train, X_val, X_test, y_train, y_val, y_test.
                                   If 'unpack' is False, returns ds_train, ds_val, ds_test.
    """
    ds = get_ds(add_features=add_features)
    if include_val:
        ds_list = slide_ds_train_val_test_split(ds)
    else:
        ds_list = slide_ds_train_test_split(ds)
    if unpack:
        unpacked_datasets = unpack_ds_list(ds_list)
        if use_pca:
            return pca_split(unpacked_datasets)
        return unpacked_datasets
    else:
        if use_pca:
            warnings.warn('PCA is not used if unpack=False', UserWarning)
        return ds_list
