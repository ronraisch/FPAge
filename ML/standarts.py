import os

AMOUNT_COL_NAME = 'amount'
SUM_COL_NAME = 'sum'
NORM_COL_NAME = 'norm'
DENOISE_COL_NAME = 'denoise'
PCA_COL_NAME = 'pca'
RANDOM_COL_NAME = 'random_feature_num'
# Constants
TARGET_LABEL = 'fp_age'
DATA_FOLDER = f'data{os.sep}ML{os.sep}MS'
DATA_PREFIX = 'slide'
RECORDS_FILE_NAME = f'data{os.sep}ML{os.sep}printing_records.xlsx'
RECORDS_SHEET_NAME = '1'
RANDOM_STATE = 420
TEST_SIZE = 0.1
VAL_SIZE = 0.1
# Define a metric choice parameter
CV_FOLDS = 5
DATASET_FILE_NAME = f'ML{os.sep}datasets{os.sep}dataset'
NOISE_FOLDER = f'data{os.sep}ML{os.sep}noise_measurements'
