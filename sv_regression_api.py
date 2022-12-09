"""
Module Docstring: regression with shapley values.
See for a brief explanation of what Shapley Value regression is:
https://www.displayr.com/shapley-value-regression/
https://stats.stackexchange.com/questions/234874/what-is-shapley-value-regression-and-how-does-one-implement-it
See for an implementation in R:
https://cran.r-project.org/web/packages/ShapleyValue/vignettes/ShapleyValue.html
https://prof.bht-berlin.de/groemping/software/relaimpo/
"""
from itertools import combinations
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from icecream import ic
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# from scipy.stats import pearsonr
# from typing import Tuple

class SvRegression():
    """This class performs linear regression using Shapley Values from game theory.
    Based on paper https://www.researchgate.net/publication/229728883_Analysis_of_Regression_in_Game_Theory_Approach
    """
    def __init__(self,
                 data=None,
                 ind_predictors_selected=None,  # predictors selected, must be a list of indices. If None, all are selected.
                 target=None):

        self.data_sv = pd.read_csv(data,
                                   index_col="date",
                                   parse_dates=True,
                                   infer_datetime_format=True)
        # Check that target is indeed in the dataset.
        assert target in self.data_sv.columns  # check that the target is in the dataset.

        # Todo: find a way more subtle to handle missing values.
        n_rows = self.data_sv.shape[0]
        self.data_sv = self.data_sv.dropna()

        n_rows_complete, n_cols = self.data_sv.shape
        print(f"{n_rows - n_rows_complete} rows have been deleted due to missing values.")
        print(f"{n_rows_complete} rows in the dataset: {data}.")
        print(f"{n_cols - 1} features (regressors) present in the dataset.")

        # Initializing features and target.
        self.x_features = np.array(self.data_sv.drop(labels=target, axis=1))
        self.y_targets = np.array(self.data_sv[target].ravel())
        # compute the number of features, to be corrected if ind_predictors_selected is not None.
        self.num_feat_selec = self.x_features.shape[1]
        self.ind_predictors_selected = list(range(self.x_features.shape[1]))

        if ind_predictors_selected is not None:
            self.ind_predictors_selected = ind_predictors_selected
            # Selecting only selected predictors.
            self.x_features = self.x_features[:, ind_predictors_selected]
            self.num_feat_selec = self.x_features.shape[1]
            print(f"{self.num_feat_selec} features selected.")

        # Scalers for features and target:
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()

        # Normalizing x_features and y_target.
        self.x_features_norm = self._scaler_x.fit_transform(self.x_features)
        self.y_targets_norm = self._scaler_y.fit_transform(self.y_targets.reshape(-1, 1))

        # Defining a linear regression object to compute r_squared.
        self.lin_reg = LinearRegression(n_jobs=-1, copy_X=False)

    def normalize(self):
        """Normalize features and targets selected using
        the class StandardScaler from Scikit-Learn.

        Returns
        -------
        x_features_norm : ndarray of shape (n_sample, n_features)
            Features normalized (each feature has zero mean and unit variance).
        y_targets_norm : ndarray of shape (n_sample, 1)
            Targets normalized (zero mean and unit variance).
        """
        x_features_norm = self._scaler_x.fit_transform(self.x_features)
        y_targets_norm = self._scaler_y.fit_transform(self.y_targets.reshape(-1, 1))
        return x_features_norm, y_targets_norm

    def unnormalize(self, x_features_norm, y_features_norm):
        """Denormalize features and targets using
        the class StandardScaler from Scikit-Learn.

        Parameters
        ----------
        x_features_norm : ndarray of shape (n_sample, n_features)
            Features normalized (each feature has zero mean and unit variance).
        y_targets_norm : ndarray of shape (n_sample, 1)
            Targets normalized (zero mean and unit variance).

        Returns
        -------
        x_features : ndarray of shape (n_sample, n_features)
            Features unnormalized.
        y_target : ndarray of shape (n_sample, 1)
            Targets unnormalized (zero mean and unit variance).
        """
        x_features = self._scaler_x.inverse_transform(x_features_norm)
        y_targets = self._scaler_y.inverse_transform(y_features_norm)
        return x_features, y_targets

    def _get_rsquared_sk(self, ind):
        """Compute a R^2 using the class LinearRegression from Scikit Learn Intelex.
        Features onto which the regression is performed are selected using a boolean
        mask obtained from the binary representation of ind.
        This method is intended for internal use only.

        Parameters
        ----------
        ind : int
            indice whose binary representation serves to compute a boolean mask
            which is used to select a given coalition of features.
            If ind = 0, R^2 = 0.0 is returned.

        Returns
        -------
        r_squared : float
            r^squared computed on the coalition of features obtained from
            the binary representation of ind.
        """
        if ind == 0:
            return 0.0
        else:
            ind_form = f"0{self.num_feat_selec}b"
            ind_bin = format(ind, ind_form)
            mask = [bool(int(ind_f)) for ind_f in ind_bin[::-1]]
            x_features_curr = self.x_features_norm[:, mask]
            self.lin_reg.fit(x_features_curr, self.y_targets_norm)
            r_squared = self.lin_reg.score(x_features_curr, self.y_targets_norm)
            return r_squared

    def compute_list_r_squared(self):
        """Compute all R^2 by performing linear regression
        on all coalitions of selected predictors.
        if n predictors are selected, 2^n coalitions are possible
        and a r^squared is evaluated on each of them.

        Warning: this function can take a long time for running
        depending on the number of predictors selected.

        Returns
        -------
        list_r_squared : list[float]
            python list of length 2^n, containing the R^2
            for all coalitions of predictors possible.
        """
        start = timer()
        list_r_squared = [self._get_rsquared_sk(ind) for ind in range(0, 2**self.num_feat_selec)]
        time_comp = timer() - start
        print(f"{round(time_comp, 2)} s to compute all r_squared.")
        return list_r_squared

    def compute_usefullness(self, predictors=None, r_squared_dum_compr=None, target=2, len_predictors=4):
        """Compute usefullness co

        Parameters
        ----------
        predictors : _type_, optional
            _description_, by default None
        r_squared_dum_compr : _type_, optional
            _description_, by default None
        target : int, optional
            _description_, by default 2
        len_predictors : int, optional
            _description_, by default 4

        Returns
        -------
        _type_
            _description_
        """
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
            predictors = [x for x in predictors if x is not target]
            bin_predictors_without_target = [1 if x in predictors else 0 for x in range(len_predictors)]
            ind_rsquared_without_target = int("".join(str(x) for x in bin_predictors_without_target), 2)
            r_squared_without_target = r_squared_dum_compr[ind_rsquared_without_target]

            return r_squared_with_target - r_squared_without_target

    def compute_shapley(self, r_squared_dum_compr=None, target_pred=None, predictors=None):
        """_summary_

        Parameters
        ----------
        r_squared_dum_compr : _type_, optional
            _description_, by default None
        target_pred : _type_, optional
            _description_, by default None
        predictors : _type_, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if r_squared_dum_compr is None:
            raise ValueError("r_squared_dum_compr cannot be None.")
        if target_pred not in predictors:
            raise ValueError(f"""\npredictors: \n{predictors}.\ntarget_pred:\n{target_pred}\n""" +
                              """target_pred must be in predictors.""")
        num_predictors = len(predictors)
        shapley_val = 0
    # First, second, third etc... term
        for len_comb in range(num_predictors):
            sum_usefullness = 0
            npfactor = np.math.factorial
            weight = (npfactor(len_comb) * npfactor(num_predictors - len_comb - 1)) / npfactor(num_predictors)

            for coalition in filter(lambda x: target_pred in x, combinations(predictors, len_comb)):
                usefullness = self.compute_usefullness(predictors=coalition,
                                                       r_squared_dum_compr=r_squared_dum_compr,
                                                       target=target_pred,
                                                       len_predictors=len(predictors))
                sum_usefullness = sum_usefullness + usefullness
            shapley_val = shapley_val + weight * sum_usefullness

        return shapley_val


    def check_norm_shap(self, predictors=None, list_r_squared=None):
        """_summary_

        Parameters
        ----------
        predictors : _type_, optional
            _description_, by default None
        list_r_squared : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if list_r_squared is None:
            raise ValueError("list_r_squared cannot be None.")
        lin_reg_fit = self.lin_reg.fit(self.x_features_norm, self.y_targets_norm)
        r_squared_full = lin_reg_fit.score(self.x_features_norm, self.y_targets_norm)
        sum_shap = 0.0

        for ind_feat in predictors:
            shap = self.compute_shapley(r_squared_dum_compr=list_r_squared, target_pred=ind_feat, predictors=predictors)
            sum_shap = sum_shap + shap

        return {"r_squared_full": r_squared_full,
                "sum_shape": sum_shap}

# Testing:
# Dataset path.
DATASET = "data/base_test_sv_reg_working.csv"

sv_reg = SvRegression(data=DATASET,
                      ind_predictors_selected=list(range(8)),
                      target="qlead_auto")

feat_norm, tar_norm = sv_reg.normalize()
feat, tar = sv_reg.unnormalize(x_features_norm=feat_norm, y_features_norm=tar_norm)
ic(feat.shape)
ic(tar.shape)

list_r_squareds = sv_reg.compute_list_r_squared()

ind_preds = sv_reg.ind_predictors_selected

dum_shap = sv_reg.compute_shapley(r_squared_dum_compr=list_r_squareds, target_pred=3, predictors=ind_preds)
ic(dum_shap)

check_norm = sv_reg.check_norm_shap(predictors=ind_preds, list_r_squared=list_r_squareds)

print(check_norm)


# Activate GPU acceleration.
# Problem: requires dpctl to work:
# https://pypi.org/project/dpctl/

# I had an issue installing dpctl, turning off for now.
# with config_context(target_offload="gpu:0"):

# r_squared_dum_compr is defined as a global variable to avoid memory overload.
# start = timer()
# r_squared_dum_compr = [get_rsquared_sk(ind) for ind in range(0, 2**dum_num)]
# time_comp = timer() - start
