# Load files for classification
import pandas as pa
from sklearn_pandas import DataFrameMapper

def extract_dataframe_from_csv(location):
    raw_data = pa.read_csv(location)
    dataset = DataFrameMapper(raw_data)
    return dataset