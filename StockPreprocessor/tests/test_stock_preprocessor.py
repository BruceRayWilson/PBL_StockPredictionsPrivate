import os
from unittest.mock import MagicMock
import pandas as pd
import pytest
# from stock_preprocessor import StockPreprocessor  # Import your StockPreprocessor class here
from StockPreprocessor.StockPreprocessor import StockPreprocessor

# Constants for tests
TEST_INPUT_DIRECTORY = 'test_data'
TEST_OUTPUT_DIRECTORY = 'test_preprocessed_data'
TEST_FILE_NAME = 'test_stock_data.csv'

# Sample data for testing
SAMPLE_STOCK_DATA = pd.DataFrame({
    'Date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
    'Open': [100, 101, 102, 103, 104],
    'High': [110, 111, 112, 113, 114],
    'Low': [90, 91, 92, 93, 94],
    'Close': [105, 106, 107, 108, 109],
    'Volume': [1000, 1100, 1200, 1300, 1400]
})

@pytest.fixture(scope='module')
def stock_preprocessor():
    # Setup the preprocessor with test directories
    preprocessor = StockPreprocessor()
    preprocessor.input_directory = TEST_INPUT_DIRECTORY
    preprocessor.output_directory = TEST_OUTPUT_DIRECTORY
    return preprocessor

@pytest.fixture(scope='module')
def stock_data_csv(tmpdir_factory):
    # Create a temporary CSV file
    df = SAMPLE_STOCK_DATA
    file = tmpdir_factory.mktemp('data').join(TEST_FILE_NAME)
    df.to_csv(file, index=False)
    return str(file)

def test_preprocess_file(mocker, stock_preprocessor, stock_data_csv):
    # Mock the os.path.join to return the test file path
    mocker.patch('os.path.join', return_value=stock_data_csv)

    # Mock the os.makedirs to prevent creating new directories
    mocker.patch('os.makedirs', return_value=None)

    # Run the preprocess_file method
    stock_preprocessor.preprocess_file(TEST_FILE_NAME)

    # Read the preprocessed file
    preprocessed_file_path = os.path.join(TEST_OUTPUT_DIRECTORY, TEST_FILE_NAME)
    preprocessed_data = pd.read_csv(preprocessed_file_path)

    # Asserts to validate the preprocessed data
    assert not preprocessed_data.isnull().values.any(), "There should be no NaN values"
    assert 'Tomorrow_Gain' in preprocessed_data.columns, "Tomorrow_Gain column should exist"
    assert 'rolling_close_2' in preprocessed_data.columns, "rolling_close_2 column should exist"
    assert 'rolling_close_5' in preprocessed_data.columns, "rolling_close_5 column should exist"
    assert 'rolling_close_60' in preprocessed_data.columns, "rolling_close_60 column should exist"

    # Clean up the preprocessed file
    os.remove(preprocessed_file_path)

# Add more tests as needed to cover edge cases, different datasets, etc.
