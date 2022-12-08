import numpy as np
import pandas as pd

from itertools import chain, combinations
from timeit import default_timer as timer
from icecream import ic
from sklearnex import patch_sklearn, config_context

patch_sklearn()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scipy.stats import pearsonr
from icecream import ic
from typing import Tuple

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
        # assert target in self.data_sv.columns

        # Todo: find a way more subtle to handle missing values.
        n_rows = self.data_sv.shape[0]

        self.data_sv = self.data_sv.dropna()
        n_rows_complete, n_cols = self.data_sv.shape
        print(f"{n_rows - n_rows_complete} rows have been deleted due to missing values.")
        print(f"{n_rows_complete} rows in the dataset: {data}.")
        print(f"{n_cols - 1} features (regressors) present.")

        # Initializing features and target.
        self.x_features = np.array(self.data_sv.drop(labels=target, axis=1))
        self.y_target = np.array(self.data_sv[target].ravel())

        # Scalers for features and target:
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()

        # Normalizing x_features and y_target.
        self.x_features_norm = self._scaler_x.fit_transform(self.x_features)
        self.y_target_norm = self._scaler_y.fit_transform(self.y_target.reshape(-1, 1))
        # print(f"max: {np.amax(self.x_features_norm)}")  # just to check normalization.

        # initializing C (sample correlations, size (n, n)) and r (sample-target correlations, size (n,)).
        self.c_jk = np.matmul(self.x_features.T, self.x_features)
        # assert self.c_jk.shape == (self.x_features.shape[1], self.x_features.shape[1])
        self.r_j = np.matmul(self.x_features.T, self.y_target)
        # assert self.r_j.shape == (self.x_features.shape[1],)

    def normalize(self):
        x_features_norm = self._scaler_x.fit_transform(self.x_features)
        y_target_norm = self._scaler_y.fit_transform(self.y_target.reshape(-1, 1))
        return x_features_norm, y_target_norm

    def unnormalize(self, x_features_norm, y_features_norm):
        x_features = self._scaler_x.inverse_transform(x_features_norm)
        y_features = self._scaler_y.inverse_transform(y_features_norm)
        return x_features, y_features

    def get_c_jk(self):
        return np.matmul(self.x_features_norm.T, self.x_features_norm)

    def get_r_j(self):
        return np.matmul(self.x_features_norm.T, self.y_target_norm)

    def get_b(self):
        c_inv = np.linalg.inv(self.get_c_jk())
        return np.matmul(c_inv, self.get_r_j())


    def get_r_squared(self, predictors: Tuple[int, ...] = None) -> float:
        n_features = self.x_features_norm.shape[1]
        if predictors is not None:
            if not all(0 <= elem <= n_features for elem in predictors):
                raise ValueError(f"All elements of predictors must be in between 0 and {n_features - 1}.")
        # Defining x_features that corresponds to the model defined by predictors:
            x_features_norm = self.x_features_norm[:, predictors]
        else:
        # We get the model with all predictors.
            x_features_norm = self.x_features_norm
        # Targets (shape=(N, 1)) do not change in either cases.
        y_target_norm = self.y_target_norm
        n_features = x_features_norm.shape[1]
        c_jk_norm = np.matmul(x_features_norm.T, x_features_norm)
        # assert c_jk_norm.shape == (n_features, n_features)
        r_j_norm = np.matmul(x_features_norm.T, y_target_norm)
        # assert r_j_norm.shape == (n_features, 1)
        b_j_norm = np.matmul(np.linalg.inv(c_jk_norm), r_j_norm)
        # assert b_j_norm.shape == (n_features, 1)
        r_squared_norm = np.matmul(b_j_norm.T, r_j_norm)

        return r_squared_norm.ravel().item()


# Testing:
# Dataset path.
dataset = "data/base_test_sv_reg_working.csv"
sv_reg = SvRegression(data=dataset,
                      target="qlead_auto")

x_features_norm, y_target_norm = sv_reg.normalize()

x_features, y_target = sv_reg.unnormalize(x_features_norm, y_target_norm)
print("shape: y_target not normalized:")
print(y_target.shape)

print("shape: y_target normalized:")
print(y_target_norm.shape)

dum_num = 5
dum_predictors = list(range(dum_num))
x_features_norm_dum = x_features_norm[:, dum_predictors]

# declaring model
lin_reg = LinearRegression(n_jobs=-1, copy_X=False)

def get_rsquared_sk(ind=None):
    if ind == 0:
        return 0.0
    else:
        # TODO: find a way to format the format code dynamically (e.g replace '5' by dum_num at runtime).
        ind_bin = f"{ind:05b}"
        dum_mask = [bool(int(ind_f)) for ind_f in ind_bin[::-1]]
        x_features_curr = x_features_norm_dum[:, dum_mask]
        lin_reg.fit(x_features_curr, y_target_norm)
        return lin_reg.score(x_features_curr, y_target_norm)

# Activate GPU acceleration.
# Problem: requires dpctl to work:
# https://pypi.org/project/dpctl/

# I had issue installing it, turning off for now.
# with config_context(target_offload="gpu:0"):

start = timer()
r_squared_dum_compr = [get_rsquared_sk(ind) for ind in range(0, 2**dum_num)]
time_comp = timer() - start

# Comptue usefullness as defined by formulae 18 from the article.
# We do not pass r_squared_dum_compr as a parameter to avoid memory overload.
def compute_usefullness(predictors=None, target=2, len_predictors=4):

    if len(predictors) == 1:
        # Rsquared corresponding to length 1 predictors:
        bin_predictors = [1 if x in predictors else 0 for x in range(len_predictors)]
        ind_rsquared = int("".join(str(x) for x in bin_predictors), 2)
        r_squared = r_squared_dum_compr[ind_rsquared]

        return r_squared

    else:
        # Rsquared with target:
        bin_predictors = [1 if x in predictors else 0 for x in range(len_predictors)]
        ind_rsquared = int("".join(str(x) for x in bin_predictors), 2)
        r_squared_with_target = r_squared_dum_compr[ind_rsquared]

        # Rsquared without target:
        # predictors.remove(target)  # bad idea to use list.remove() --> add side effects to the function.
        predictors = [x for x in predictors if x is not target]
        bin_predictors_without_target = [1 if x in predictors else 0 for x in range(len_predictors)]
        ind_rsquared_without_target = int("".join(str(x) for x in bin_predictors_without_target), 2)
        r_squared_with_target_without_target = r_squared_dum_compr[ind_rsquared_without_target]

        return r_squared_with_target - r_squared_with_target_without_target

dum_us = compute_usefullness(predictors=[0, 2, 3], target=2, len_predictors=dum_num)


# Test Case: Computing Shapley value for the second predictor.
dum_target = 2
ic(dum_predictors)

def compute_shapley(target_pred=None, predictors=None):
    num_predictors = len(predictors)
    shapley_val = 0
# First, second, third etc... term
    for len_comb in range(num_predictors):
        sum_usefullness = 0
        weight = (np.math.factorial(len_comb) * np.math.factorial(num_predictors - len_comb - 1)) / np.math.factorial(num_predictors)
        for coalition in filter(lambda x: target_pred in x, combinations(predictors, len_comb)):
            # coalition = list(coalition)  # convert to list to make it mutable, need to remove target (see function below).
            usefullness = compute_usefullness(predictors=coalition,
                                              target=target_pred,
                                              len_predictors=len(predictors))
            sum_usefullness = sum_usefullness + usefullness
        shapley_val = shapley_val + weight * sum_usefullness

    return shapley_val

dum_shap = compute_shapley(target_pred=2, predictors=[1,2,3,4])
ic(dum_shap)

# ic(r_squared_dum_compr)
# ic(dum_us)
# ic(time_comp)



