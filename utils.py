import numpy as np
import pandas as pd
from icecream import ic

path_dataset = "data/base_test_sv_reg_working.csv"
data_sv = pd.read_csv(path_dataset,
                      index_col="date",
                      parse_dates=True,
                      infer_datetime_format=True)

# Dropping irrelevant "Segment" column:
# data_sv = data_sv.drop(labels="Segment", axis=1)
# Normalizing columns names:
data_sv.columns = data_sv.columns.str.strip().str.replace(" ", "_")
data_sv.columns = data_sv.columns.str.lower()

x_features = data_sv.drop(labels="qlead_auto", axis=1)
y_target = data_sv["qlead_auto"]

nb_rows, nb_features = x_features.shape
print(f"{nb_rows} rows in the dataset.")
print(f"{nb_features} features in the dataset.")

