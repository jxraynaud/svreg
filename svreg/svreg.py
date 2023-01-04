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

import numpy as np
import pandas as pd
from alive_progress import alive_bar
from scipy.stats import pearsonr
from sklearnex import patch_sklearn

patch_sklearn()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class SvRegression:
    """This class performs linear regression using Shapley Values from game theory.
    Based on paper https://www.researchgate.net/publication/229728883_Analysis_of_Regression_in_Game_Theory_Approach
    """

    def __init__(
        self,
        data=None,  # Must be a dataframe or an array (not a path to a file).
        ind_predictors_selected=None,  # predictors selected, must be a list of indices. If None, all are selected.
        target=None,
    ):

        self._data_sv = data
        # Check that target is indeed in the dataset.
        if target not in self._data_sv.columns:
            raise ValueError(f"{target} not in the dataset.")

        # Todo: find a way more subtle to handle missing values.
        n_rows = self._data_sv.shape[0]
        self._data_sv = self._data_sv.dropna()

        n_rows_complete, n_cols = self._data_sv.shape

        print(f"{n_rows - n_rows_complete} rows have been deleted due to missing values.")
        print(f"{n_rows_complete} rows in the dataset.")
        print(f"{n_cols - 1} features (regressors) present in the dataset.")

        # Initializing features and target.
        self._data_without_target = self._data_sv.drop(labels=target, axis=1)
        self.x_features = np.array(self._data_without_target)
        self.y_targets = np.array(self._data_sv[target].ravel())
        # compute the number of features, to be corrected if ind_predictors_selected is not None.
        self.num_feat_selec = self.x_features.shape[1]
        self.ind_predictors_selected = list(range(self.num_feat_selec))

        if ind_predictors_selected is not None:
            self.ind_predictors_selected = ind_predictors_selected
            # Selecting only selected predictors.
            self.x_features = self.x_features[:, ind_predictors_selected]
            self.num_feat_selec = self.x_features.shape[1]
            print(f"{self.num_feat_selec} features selected.")
        self.names = self._data_without_target.columns[self.ind_predictors_selected]
        # Scalers for features and target:
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()

        # Normalizing x_features and y_target.
        self.x_features_norm = self._scaler_x.fit_transform(self.x_features)
        self.y_targets_norm = self._scaler_y.fit_transform(self.y_targets.reshape(-1, 1))

        # Defining a linear regression object to compute r_squared.
        self.lin_reg = LinearRegression(n_jobs=-1, copy_X=False)

        # empty list.
        self._list_r_squared = None

        # initializing coefficients array (unnormalized regressions coeffs).
        self.coeffs = np.zeros(self.num_feat_selec)
        # initializing coefficients array (normalized regressions coeffs).
        self.coeffs_norm = np.zeros(self.num_feat_selec)
        # initializing Shapley values array (normalized basis).
        self.shaps = np.zeros(self.num_feat_selec)

    @property
    def data_sv(self):
        """Setter for the private _data_sv dataframe.
        No setter should be written for this property,
        making it a defacto "readonly" variable.

        Returns:
          _data_sv : pandas dataframe
          data used for the regression.
        """

        return self._data_sv

    @property
    def list_r_squared(self):
        """Setter for the list_r_squared list.
        No setter should be written for this property,
        making it a defacto "readonly" variable.

        Returns:
            self._list_r_squared : python list of length 2^nfeatures.
            Contains the R^2 of regressions computed from the differents
            coalitions of features.
        """

        if self._list_r_squared is None:
            num_r_squareds = 2**self.num_feat_selec
            print(f"Computing the {num_r_squareds} linears regressions.")
            with alive_bar(num_r_squareds, title="Linear regressions") as bar_:
                self._list_r_squared = [self._get_rsquared_sk(ind, bar_=bar_) for ind in range(0, num_r_squareds)]
        return self._list_r_squared

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

    def _get_rsquared_sk(self, ind, bar_=None):
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
        bar_ : context manager from alive_progress.
            Context manager used to track the progresses of the alive progress bar.

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
            # Update the progress bar after each linear regression computation.
            if bar_ is not None:
                bar_()
            return r_squared

    def compute_usefullness(self, coalition, target=2):
        """Compute the usefulness corresponding to the coalition
        of predictors "coalition" with the target predictor "target".

        Parameters
        ----------
        coalition : list[int]
            list of indices of predictors in the current coalition.
        target : int, optional
            index of the predictor whose usefullness if computed
            for the given coalition, by default 2.

        Returns
        -------
        usefullness : float
            usefullness computed on the coalition "coalition" with target "target".
        """

        len_predictors = self.num_feat_selec

        if len(coalition) == 1:
            # Rsquared corresponding to length 1 predictors:
            bin_coalition = [1 if x in coalition else 0 for x in range(len_predictors)]
            ind_rsquared = int("".join(str(x) for x in bin_coalition), 2)
            r_squared = self.list_r_squared[ind_rsquared]

            return r_squared

        else:
            # Rsquared with target:
            bin_coalition_with_target = [1 if x in coalition else 0 for x in range(len_predictors)]
            ind_rsquared = int("".join(str(x) for x in bin_coalition_with_target), 2)
            r_squared_with_target = self.list_r_squared[ind_rsquared]

            # Rsquared without target:
            coalition = [x for x in coalition if x is not target]
            bin_coalition_without_target = [1 if x in coalition else 0 for x in range(len_predictors)]
            ind_rsquared_without_target = int("".join(str(x) for x in bin_coalition_without_target), 2)
            r_squared_without_target = self.list_r_squared[ind_rsquared_without_target]

            # Getting usefullness:
            usefullness = r_squared_with_target - r_squared_without_target

            return usefullness

    def compute_shapley(self, ind_feat=2):
        """Compute shapley value using target_pred as the indice
        of the predictor of interest.

        Parameters
        ----------
        ind_feat : int
            index of the predictor of interest of the shapley value
            to be computed, by default 2

        Returns
        -------
        shapley_val : float
            shapley value computed from the differents coalitions
            of predictors that contain "target_pred".

        Raises
        ------
        ValueError
            if list_r_squared is None.
        ValueError
            if target_pred is not in self.ind_predictors_selected.
        """

        num_predictors = self.num_feat_selec
        predictors = self.ind_predictors_selected
        assert ind_feat in self.ind_predictors_selected, "target_pred must be in predictors"
        feat = predictors[ind_feat]
        if self.list_r_squared is None:
            raise ValueError("list_r_squared cannot be None.")
        # Commenting for now.
        # TODO: find a clever way to raise this error.
        # Initializing shapley value to 0.0.
        shapley_val = 0
        npfactor = np.math.factorial
        # First, second, third etc... term
        for len_comb in range(0, num_predictors + 1):
            sum_usefullness = 0
            # Trick to shift weigths correctly.
            # For 0 length coalition, we enforce weight = 0 and then,
            # for the remaining terms we compute the weigths by shifting backward
            # len_comb.
            if len_comb == 0:
                weight = 0
            else:
                len_comb_int = len_comb - 1
                weight = (npfactor(len_comb_int) * npfactor(num_predictors - len_comb_int - 1)) / npfactor(num_predictors)

            for coalition in filter(lambda x: feat in x, combinations(predictors, len_comb)):
                usefullness = self.compute_usefullness(coalition=coalition, target=feat)
                sum_usefullness = sum_usefullness + usefullness
            shapley_val = shapley_val + weight * sum_usefullness
        return shapley_val

    def fit(self):
        """Compute the coefficients of regression ajusted
        using Shapley Values.

        Returns
        -------
        self.coeffs: numpy array of shape (self.num_feat_selec + 1,)
            The first element of self.coeffs is the intercept term
            in the unnormalized basis.
            The remaining elements are the coefficients of regressions for each predictor.
        """

        target = self.y_targets_norm.ravel()

        for ind_feat in range(0, self.num_feat_selec):
            curr_shap = self.compute_shapley(ind_feat=ind_feat)
            self.coeffs[ind_feat] = curr_shap
            self.shaps[ind_feat] = curr_shap
            # Normalizing the shapley value with the correlation coefficient between
            # the current feature and the target.
            curr_feat = self.x_features_norm[:, ind_feat]
            # TODO: take the p-value into account to test the significance
            # of the correlation.
            corr = pearsonr(curr_feat, target).statistic
            self.coeffs[ind_feat] = self.coeffs[ind_feat] / corr

        # Saving unnormalized coefficients:
        self.coeffs_norm = np.copy(self.coeffs)
        # Unnormalize coefficients of the Shapley Value regression.
        self._unnormalize_coeffs()

        return self.coeffs

    def _unnormalize_coeffs(self):
        """Unnormalize the coefficients of regressions
        for each selected predictors.
        Compute as well the intercept term
        in the unnormalized basis.

        Returns
        -------
        self.coeffs: numpy array of shape (self.num_feat_selec + 1,)
            The first element of self.coeffs is the intercept term
            in the unnormalized basis.
            The remaining elements are the coefficients of regressions for each predictor.
        """
        means_x = self._scaler_x.mean_
        means_y = self._scaler_y.mean_

        stds_x = np.sqrt(self._scaler_x.var_)
        stds_y = np.sqrt(self._scaler_y.var_)

        self.coeffs = (self.coeffs * stds_y) / stds_x
        offset = means_y - ((means_x * stds_y) / stds_x).sum()

        self.coeffs = np.concatenate((offset, self.coeffs))

        return self.coeffs

    def check_norm_shap(self):
        """Compute both R^2 of the full model (all predictors)
        and the sum of shapley values.
        The two should be equal in the normalized basis
        (see eq. 19 of the paper cited in module docstring).

        Returns
        -------
        Python dictionnary
            Contains the R^2 of the full model (r_squared_full)
            and the sum of shapley values (sum_shaps)

        Raises
        ------
        ValueError
            if list_r_squared is None.
        """

        if self.list_r_squared is None:
            raise ValueError("list_r_squared cannot be None.")
        lin_reg_fit = self.lin_reg.fit(self.x_features_norm, self.y_targets_norm)
        r_squared_full = lin_reg_fit.score(self.x_features_norm, self.y_targets_norm)

        sum_shap = 0.0
        predictors = self.ind_predictors_selected
        for ind_feat in predictors:
            shap = self.compute_shapley(ind_feat=ind_feat)
            sum_shap = sum_shap + shap
        return {"r_squared_full": r_squared_full, "sum_shaps": sum_shap}

    def histo_shaps(self):
        """Plot the histogram of the shapley values.

        Returns
        -------
        None
        """
        shaps = np.sort(np.abs(self.shaps))[::-1]
        plt.figure(figsize=(10, 5))
        plt.bar(self.names, shaps)
        plt.xlabel("Features")
        plt.ylabel("Shapley values")
        plt.title("Histogram of Shapley values")
        plt.show()


if __name__ == "__main__":

    # Testing:
    DATASET = "../data/mtcars.csv"
    df_dataset = pd.read_csv(DATASET, index_col="model")

    sv_reg = SvRegression(
        data=df_dataset,
        ind_predictors_selected=list(range(10)),
        target="mpg",
    )

    # Fitting the regression.
    coeffs = sv_reg.fit()

    print("=" * 70)
    print("Per predictor Shapley value (normalized basis).")
    print(sv_reg.shaps)
    print("=" * 70)
    print("Coefficients of the SV regression (normalized basis).")
    print(sv_reg.coeffs_norm)
    print("=" * 70)
    print("Coefficients of the SV regression (unnormalized basis).")
    print("sv_reg.coeffs[0] --> intercept term.")
    print(sv_reg.coeffs)
    print("=" * 70)
    print("Checking that the Shapley Values sums up to the full model R^2.")
    print(sv_reg.check_norm_shap())
    print("=" * 70)
