from glob import glob

import numpy as np
import pandas as pd

from ML.preprocess.parse.raw_data import parse_csv_file, interpolate_spectrums
from ML.standarts import AMOUNT_COL_NAME, NORM_COL_NAME, DENOISE_COL_NAME, SUM_COL_NAME
from ML.standarts import NOISE_FOLDER


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


def add_normalized_amount_column(ds):
    X = ds[get_columns_prefix(ds)]
    X_normalized = X.div(X.sum(axis=1), axis=0)
    norm_cols = [f'{NORM_COL_NAME}_{col_name[len(AMOUNT_COL_NAME) + 1:]}' for col_name in X.columns]
    X_normalized.columns = norm_cols
    ds = pd.concat([ds, X_normalized], axis=1)
    return ds


def load_noise_spectrums():
    noise_spectrum_list = [parse_csv_file(file_name) for file_name in glob(f'{NOISE_FOLDER}/*.csv')]
    return noise_spectrum_list


def preprocess_noise_spectrums(noise_spectrum_list):
    norm_noise_spectrum_list = [noise_spectrum / noise_spectrum.sum() for noise_spectrum in
                                noise_spectrum_list]
    norm_noise_spectrum_df = interpolate_spectrums(norm_noise_spectrum_list).drop(['file_name'],
                                                                                  axis=1)
    return norm_noise_spectrum_df


def get_mean_noise_spectrum(norm_noise_spectrum_df):
    return norm_noise_spectrum_df.mean(axis=0)


def remove_col_prefix(x):
    x.index = [f'{index_name[len(NORM_COL_NAME) + 1:]}' for index_name in x.index]
    x.index = x.index.astype(np.float64)
    return x


def interpolate_noise(x, mean_noise_spectrum):
    x = remove_col_prefix(x)
    noise_inter = interpolate_spectrums([x, mean_noise_spectrum])
    noise_inter = noise_inter[x.index].values[1, :]
    return noise_inter


def denoise(X, noise_inter, epsilon=1e-6):
    covariance = np.cov(X, noise_inter)
    C = covariance[-1, :-1] / (noise_inter.std() ** 2 + epsilon)
    X_denoised = X - np.matmul(C.reshape((-1, 1)), noise_inter.reshape((1, -1)))
    X_denoised[X_denoised <= 0] = 0
    return X_denoised


def add_denoise_columns(ds):
    X = ds[get_columns_prefix(ds, NORM_COL_NAME)]

    noise_spectrum_list = load_noise_spectrums()
    norm_noise_spectrum_df = preprocess_noise_spectrums(noise_spectrum_list)
    mean_noise_spectrum = get_mean_noise_spectrum(norm_noise_spectrum_df)
    x = X.iloc[0]
    noise_inter = interpolate_noise(x, mean_noise_spectrum)
    X_denoised = denoise(X, noise_inter)
    denoise_cols = [f'{DENOISE_COL_NAME}_{col_name[len(NORM_COL_NAME) + 1:]}' for col_name in
                    X.columns]
    X_denoised.columns = denoise_cols
    ds = pd.concat([ds, X_denoised], axis=1)
    return ds


def add_sum_col(ds):
    X = ds[get_columns_prefix(ds)]
    X_sum = X.sum(axis=1)
    X_sum = pd.DataFrame(X_sum, columns=[SUM_COL_NAME])
    ds = pd.concat([ds, X_sum], axis=1)
    return ds


def main(ds):
    ds = add_normalized_amount_column(ds)
    ds = add_denoise_columns(ds)
    ds = add_sum_col(ds)
    ds = ds.drop(get_columns_prefix(ds), axis=1)
    return ds
