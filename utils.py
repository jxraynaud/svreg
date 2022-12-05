import numpy as np
import pandas as pd

from icecream import ic
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from icecream import ic


# path_dataset = "data/base_test_sv_reg_working.csv"
# data_sv = pd.read_csv(path_dataset,
#                       index_col="date",
#                       parse_dates=True,
#                       infer_datetime_format=True)

# # Dropping irrelevant "Segment" column:
# # data_sv = data_sv.drop(labels="Segment", axis=1)
# # Normalizing columns names:
# data_sv.columns = data_sv.columns.str.strip().str.replace(" ", "_")
# data_sv.columns = data_sv.columns.str.lower()

# x_features = data_sv.drop(labels="qlead_auto", axis=1)
# y_target = data_sv["qlead_auto"]

# nb_rows, nb_features = x_features.shape
# print(f"{nb_rows} rows in the dataset.")
# print(f"{nb_features} features in the dataset.")


class SvRegression():

    """This class performs linear regression using Shapley Values from game theory.
    Based on paper https://www.researchgate.net/publication/229728883_Analysis_of_Regression_in_Game_Theory_Approach
    """
    def __init__(self,
                 data=None,
                 target=None,
                 tol_missing=2):

        self.data_sv = pd.read_csv(data,
                                   index_col="date",
                                   parse_dates=True,
                                   infer_datetime_format=True)
        # Check that target is indeed in the dataset.
        assert target in self.data_sv.columns

        # Todo: find a way more subtle to handle missing values.
        n_rows = self.data_sv.shape[0]

        self.data_sv = self.data_sv.dropna()
        n_rows_complete, n_cols = self.data_sv.shape
        print(f"{n_rows - n_rows_complete} rows have been deleted due to missing values.")
        print(f"{n_rows_complete} rows in the dataset: {data}.")
        print(f"{n_cols - 1} features (regressors) present.")

        # Initializing features and target.
        self.x_features = self.data_sv.drop(labels=target, axis=1)
        self.y_target = self.data_sv[target].ravel()

        # Scalers for features and target:
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()


    def normalize(self):
        x_features_norm = self._scaler_x.fit_transform(self.x_features)
        y_target_norm = self._scaler_y.fit_transform(self.y_target.reshape(-1, 1))
        return x_features_norm, y_target_norm

    def unnormalize(self, x_features_norm, y_features_norm):
        x_features = self._scaler_x.inverse_transform(x_features_norm)
        y_features = self._scaler_y.inverse_transform(y_features_norm)
        return x_features, y_features


# Testing:
dataset = "data/base_test_sv_reg_working.csv"
sv_reg = SvRegression(data=dataset,
                      target="qlead_auto")

x_features_norm, y_target_norm = sv_reg.normalize()
x_features, y_target = sv_reg.unnormalize(x_features_norm, y_target_norm)
print("shape: y_target not normalized:")
print(y_target.shape)

print("shape: y_target normalized:")
print(y_target_norm.shape)