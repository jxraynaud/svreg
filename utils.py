import numpy as np
import pandas as pd
from icecream import ic

path_dataset = "data/base_test_sv_reg.csv"
data_sv = pd.read_csv(path_dataset,
                      index_col="date",
                      parse_dates=True,
                      infer_datetime_format=True)

# Dropping irrelevant "Segment" column:
data_sv = data_sv.drop(labels="Segment", axis=1)
# Remove





# print(data_sv.head())
