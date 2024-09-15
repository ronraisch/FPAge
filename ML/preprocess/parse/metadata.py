import os

import numpy as np
from pandas import read_excel as pd_read_excel
from pandas import to_datetime as pd_to_datetime
from ML.standarts import DATA_FOLDER, DATA_PREFIX, RECORDS_FILE_NAME, RECORDS_SHEET_NAME


# Function to load metadata from Excel file
def load_metadata_file(file_name=RECORDS_FILE_NAME,
                       sheet_name=RECORDS_SHEET_NAME):
    """
    Load metadata from an Excel file into a Pandas DataFrame.

    Parameters:
    - file_name (str): The name of the Excel file.
    - sheet_name (str): The name of the Excel sheet containing the data.

    Returns:
    - pd.DataFrame: The loaded metadata.
    """
    return pd_read_excel(file_name, sheet_name=sheet_name)


# Function to drop unwanted columns from the records
def drop_unwanted_columns(records):
    """
    Drop unwanted columns from the records DataFrame.

    Parameters:
    - records (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with unwanted columns removed.
    """
    # Drop specific columns and remove rows with missing values
    records = records.drop(['sample index'], axis=1).dropna()
    return records


# Function to rename columns of the records
def rename_records_columns(records):
    """
    Rename columns of the records DataFrame for better readability.

    Parameters:
    - records (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with renamed columns.
    """
    records = records.rename(
        columns={'finger index': 'fp_idx', 'slide index': f'{DATA_PREFIX}_idx',
                 'imprint date': 'imprint_date', 'imprint time': 'imprint_hour',
                 'sample date': 'sample_date', 'sample time': 'sample_hour',
                 'donor\'s age': 'donor_age'})
    return records


# Function to set the data types of columns in the records
def typing_columns(records):
    """
    Set the data types of columns in the records DataFrame.

    Parameters:
    - records (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with updated data types.
    """
    records[f'{DATA_PREFIX}_idx'] = records[f'{DATA_PREFIX}_idx'].astype(
        np.int64)
    time_columns = ['sample_date', 'sample_hour', 'imprint_date',
                    'imprint_hour']
    records[time_columns] = records[time_columns].astype(str)
    return records


# Function to create a datetime column for imprint time
def create_imprint_time(records):
    """
    Create a datetime column for imprint time in the records DataFrame.

    Parameters:
    - records (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with an additional column for imprint time.
    """
    records['imprint_full_time'] = pd_to_datetime(
        records['imprint_date'] + " " + records['imprint_hour'], dayfirst=True, format='mixed')
    return records


# Function to create a datetime column for sample time
def create_sample_time(records):
    """
    Create a datetime column for sample time in the records DataFrame.

    Parameters:
    - records (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with an additional column for sample time.
    """
    records['sample_full_time'] = pd_to_datetime(
        records['sample_date'] + " " + records['sample_hour'], dayfirst=True, format='mixed')
    return records


# Function to calculate fingerprint age and remove unnecessary columns
def create_fp_age(records):
    """
    Calculate fingerprint age and remove unnecessary columns in the records DataFrame.

    Parameters:
    - records (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with fingerprint age and unnecessary columns removed.
    """
    records['fp_age'] = (records['sample_full_time'] - records[
        'imprint_full_time']).dt.total_seconds() / (3600 * 24)
    records.drop(['imprint_date', 'imprint_hour', 'sample_date', 'sample_hour',
                  'imprint_full_time', 'sample_full_time'], axis=1,
                 inplace=True)
    return records


# Function to create a file name column
def create_file_name(records):
    """
    Create a file name column in the records DataFrame.

    Parameters:
    - records (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with an additional column for file names.
    """
    records['file_name'] = DATA_FOLDER + os.path.sep + DATA_PREFIX + records[
        f'{DATA_PREFIX}_idx'].astype(str) + '_' + records['fp_idx'].astype(
        str) + '.csv'
    return records


# Function to remove bad slides from the records
def remove_bad_slides(records, bad_slides=[1]):
    """
    Remove bad slides from the records DataFrame.

    Parameters:
    - records (pd.DataFrame): The input DataFrame.
    - bad_slides (list): List of slide indices to be removed.

    Returns:
    - pd.DataFrame: The DataFrame with bad slides removed.
    """
    records = records[~records[f'{DATA_PREFIX}_idx'].isin(bad_slides)]
    return records


# Function to get processed metadata
def main():
    """
    Get processed metadata by applying a series of transformations to the original data.

    Returns:
    - pd.DataFrame: The processed metadata.
    """
    # Load metadata
    records = load_metadata_file()

    # Data preprocessing steps
    records = drop_unwanted_columns(records)
    records = rename_records_columns(records)
    records = typing_columns(records)
    records = create_imprint_time(records)
    records = create_sample_time(records)
    records = create_fp_age(records)
    records = create_file_name(records)
    records = remove_bad_slides(records)
    # records = records[records.slide_idx < 77]

    return records
