import pytest

from sv_regression.sv_regression import SvRegression


def test_init():
    dataset = "data/mtcars.csv"
    sv_reg = SvRegression(data=dataset, target="mpg")
    assert sv_reg.x_features.shape == (643, 27)


def test_init_invalid():
    dataset = "data/mtcars.csv"
    with pytest.raises(ValueError):
        _ = SvRegression(data=dataset, target="invalid_target")


def test_normalize():
    dataset = "data/mtcars.csv"
    sv_reg = SvRegression(data=dataset, target="mpg")
    x_features_norm, _ = sv_reg.normalize()
    assert x_features_norm.shape == (643, 27)


def test_unnormalize():
    dataset = "data/mtcars.csv"
    sv_reg = SvRegression(data=dataset, target="mpg")
    x_features_norm, y_target_norm = sv_reg.normalize()
    x_features, _ = sv_reg.unnormalize(x_features_norm, y_target_norm)
    assert x_features.shape == (643, 27)


def test_get_rsquared():
    dataset = "data/mtcars.csv"
    sv_reg = SvRegression(data=dataset, target="mpg")
    rsquared = sv_reg._get_rsquared_sk(sv_reg.num_feat_selec)
    assert round(rsquared, 3) == 0.885


def test_get_rsquared_ind_0():
    dataset = "data/mtcars.csv"
    sv_reg = SvRegression(data=dataset, target="mpg")
    rsquared = sv_reg._get_rsquared_sk(0)
    assert rsquared == 0


def test_compute_shapley_incorrect_target():
    dataset = "data/mtcars.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(
            data=dataset, target="incorrect_t", ind_predictors_selected=[0, 1, 2, 3, 4]
        )
        _ = sv_reg.compute_shapley(ind_feat=6)


def test_compute_shapley_1_feature(cache_compute_1_feature):
    shapley = cache_compute_1_feature
    assert round(shapley, 2) == 0.02


def test_compute_shapley_5_features(cache_compute_5_features):
    shapley = cache_compute_5_features
    assert round(shapley, 3) == 0.128


def test_compute_shapley_none():
    dataset = "data/mtcars.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(
            data=dataset, target="mpg", ind_predictors_selected=[]
        )
        _ = sv_reg.compute_shapley()


def test_compute_shapley_incorrect():
    dataset = "data/mtcars.csv"
    sv_reg = SvRegression(
        data=dataset, target="mpg", ind_predictors_selected=[0]
    )
    sv_reg._list_r_squared = None
    with pytest.raises(IndexError):
        _ = sv_reg.compute_shapley(ind_feat=5)


def test_check_norm_shap(cache_norm_shap):
    test_dict = cache_norm_shap
    assert test_dict["r_squared_full"] == test_dict["sum_shaps"]


def test_data_sv():
    dataset = "data/mtcars.csv"
    sv_reg = SvRegression(
        data=dataset, target="mpg", ind_predictors_selected=[0, 1, 2]
    )
    data = sv_reg.data_sv
    assert data.shape == (643, 28)


def test_fit():
    dataset = "data/mtcars.csv"
    sv_reg = SvRegression(
        data=dataset, target="mpg", ind_predictors_selected=[0, 1, 2]
    )
    sv_reg.fit()
    assert sv_reg.x_features.shape == (643, 3)


def test_fit_incorrect_target():
    dataset = "data/mtcars.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(
            data=dataset, target="incorrect_t", ind_predictors_selected=[0, 1, 2]
        )
        sv_reg.fit()


def test_fit_incorrect_ind():
    dataset = "data/mtcars.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(
            data=dataset, target="mpg", ind_predictors_selected=[]
        )
        sv_reg.fit()

