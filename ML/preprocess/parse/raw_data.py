import numpy as np
import pandas as pd
from ML.standarts import AMOUNT_COL_NAME

data_folder = 'data'
data_prefix = 'slide'
data_file_extension = '.csv'


def parse_csv_file(file_name):
    df = pd.read_csv(file_name)
    df = df.drop([' DriftTime', ' Analyte'], axis=1)
    new_columns = [col[1:] if col[0] == ' ' else col for col in df.columns]
    df.columns = new_columns
    df = df.drop(['Max Intensity', 'Min Intensity', 'Sum Intensity'], axis=1)
    df = df.set_index('M/z')
    df.index = df.index.astype(np.float64)
    df = df.sort_values('M/z')
    df.name = file_name
    df.columns = [file_name]
    return df[df.columns[0]]


def interpolate_spectrums(df_list):
    df = pd.DataFrame(df_list).T.sort_index()

    # Define a custom grouping function
    def custom_grouping(index):
        return round(index, 1)

    # Apply the grouping function to create a new column for grouping
    df['group'] = df.index.map(custom_grouping)

    # Group by the new 'group' column and sum the values within each group
    grouped_df = df.groupby('group').sum()
    # grouped_df.index.name='M/z'
    grouped_df = grouped_df.T.reset_index()
    grouped_df = grouped_df.rename(columns={'index': 'file_name'})
    return grouped_df


def main(records_files_names):
    raw_data = [parse_csv_file(file_name) for file_name in records_files_names]
    inter_data = interpolate_spectrums(raw_data)
    inter_data.columns = ['file_name'] + list([f'{AMOUNT_COL_NAME}_{col_name}' for col_name in
                                               inter_data.columns[1:]])
    return inter_data
