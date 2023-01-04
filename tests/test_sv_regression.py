import pytest
import numpy as np

from svreg.svreg import SvRegression


def test_init(dataset):
    #dataset = "data/mtcars.csv"
    sv_reg = SvRegression(data=dataset, target="mpg")
    assert sv_reg.x_features.shape == (32, 10)


def test_init_invalid(dataset):
    with pytest.raises(ValueError):
        _ = SvRegression(data=dataset, target="invalid_target")


def test_normalize(dataset):
    sv_reg = SvRegression(data=dataset, target="mpg")
    x_features_norm, _ = sv_reg.normalize()
    assert pytest.approx(np.amax(x_features_norm), 0.2) == 3.26
    assert pytest.approx(np.amin(x_features_norm), 0.2) == -1.90


def test_unnormalize(dataset):
    sv_reg = SvRegression(data=dataset, target="mpg")
    x_features_norm, y_target_norm = sv_reg.normalize()
    x_features, _ = sv_reg.unnormalize(x_features_norm, y_target_norm)
    assert x_features.shape == (32, 10)


def test_get_rsquared(dataset):
    sv_reg = SvRegression(data=dataset, target="mpg")
    rsquared = sv_reg._get_rsquared_sk(sv_reg.num_feat_selec)
    assert round(rsquared, 3) == 0.731


def test_get_rsquared_ind_0(dataset):
    sv_reg = SvRegression(data=dataset, target="mpg")
    rsquared = sv_reg._get_rsquared_sk(0)
    assert rsquared == 0


def test_compute_shapley_incorrect_target(dataset):
    with pytest.raises(ValueError):
        sv_reg = SvRegression(
            data=dataset, target="incorrect_t", ind_predictors_selected=[0, 1, 2, 3, 4]
        )
        _ = sv_reg.compute_shapley(ind_feat=6)


def test_compute_shapley_1_feature(dataset):
    with pytest.raises(AssertionError):
        sv_reg = SvRegression(
            data=dataset,
            target="mpg",
            ind_predictors_selected=[0]
        )
        _ = sv_reg.compute_shapley()


def test_compute_shapley_5_features(cache_compute_5_features):
    shapley = cache_compute_5_features
    assert pytest.approx(shapley, 3) == 0.022


def test_compute_shapley_none(dataset):
    with pytest.raises(ValueError):
        sv_reg = SvRegression(
            data=dataset, target="mpg", ind_predictors_selected=[]
        )
        _ = sv_reg.compute_shapley()


def test_compute_shapley_incorrect(dataset):
    sv_reg = SvRegression(
        data=dataset, target="mpg", ind_predictors_selected=[0]
    )
    sv_reg._list_r_squared = None
    with pytest.raises(Exception):
        _ = sv_reg.compute_shapley(ind_feat=5)


def test_check_norm_shap(cache_norm_shap):
    test_dict = cache_norm_shap
    assert pytest.approx(test_dict["r_squared_full"], 3)  == \
           pytest.approx(test_dict["sum_shaps"], 3)


def test_data_sv(dataset):
    sv_reg = SvRegression(
        data=dataset, target="mpg", ind_predictors_selected=[0, 1, 2]
    )
    data = sv_reg.data_sv
    assert data.shape == (32, 11)


def test_fit(dataset):
    sv_reg = SvRegression(
        data=dataset, target="mpg", ind_predictors_selected=[0, 1, 2]
    )
    sv_reg.fit()
    assert sv_reg.x_features.shape == (32, 3)


def test_fit_incorrect_target(dataset):
    with pytest.raises(ValueError):
        sv_reg = SvRegression(
            data=dataset, target="incorrect_t", ind_predictors_selected=[0, 1, 2]
        )
        sv_reg.fit()


def test_fit_incorrect_ind(dataset):
    with pytest.raises(ValueError):
        sv_reg = SvRegression(
            data=dataset, target="mpg", ind_predictors_selected=[]
        )
        sv_reg.fit()
