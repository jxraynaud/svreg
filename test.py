from sv_regression import SvRegression, LinearRegression, StandardScaler
import pytest

def test_init():
    dataset = "data/base_test_sv_reg_working.csv"
    sv_reg = SvRegression(data=dataset, target="qlead_auto")
    assert sv_reg.x_features.shape == (643, 27)

def test_init_2():
    dataset = "data/base_test_sv_reg_working.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(data=dataset, target="invalid_target")

def test_normalize():
    dataset = "data/base_test_sv_reg_working.csv"
    sv_reg = SvRegression(data=dataset, target="qlead_auto")
    x_features_norm, y_target_norm = sv_reg.normalize()
    assert x_features_norm.shape == (643, 27)


def test_unnormalize():
    dataset = "data/base_test_sv_reg_working.csv"
    sv_reg = SvRegression(data=dataset, target="qlead_auto")
    x_features_norm, y_target_norm = sv_reg.normalize()
    x_features, y_target = sv_reg.unnormalize(x_features_norm, y_target_norm)
    assert x_features.shape == (643, 27)

def test_get_rsquared():
    dataset = "data/base_test_sv_reg_working.csv"
    sv_reg = SvRegression(data=dataset, target="qlead_auto")
    rsquared = sv_reg._get_rsquared_sk(sv_reg.num_feat_selec)
    assert round(rsquared, 3) == 0.885

def test_get_rsquared_2():
    dataset = "data/base_test_sv_reg_working.csv"
    sv_reg = SvRegression(data=dataset, target="qlead_auto")
    rsquared = sv_reg._get_rsquared_sk(0)
    assert rsquared == 0

def test_compute_shapley_incorrect_target():
    dataset = "data/base_test_sv_reg_working.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(data=dataset, target="incorrect_t", ind_predictors_selected=[0, 1, 2, 3, 4])
        shapley = sv_reg.compute_shapley(ind_feat=6)

def test_compute_shapley_1_feature(cache_compute_1_feature):
    shapley = cache_compute_1_feature
    assert round(shapley, 2) == 0.02

def test_compute_shapley_5_features(cache_compute_5_features):
    shapley = cache_compute_5_features
    assert round(shapley, 3) == 0.022

def test_compute_shapley_10_features(cache_compute_10_features):
    shapley = cache_compute_10_features
    assert round(shapley, 3) == 0.022

def test_compute_shapley_15_features(cache_compute_15_features):
    shapley = cache_compute_15_features
    assert round(shapley, 3) == 0.132


def test_compute_shapley_none():
    dataset = "data/base_test_sv_reg_working.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[])
        shapley = sv_reg.compute_shapley()

def test_compute_shapley_incorrect():
    dataset = "data/base_test_sv_reg_working.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0])
        sv_reg._list_r_squared = None
        shapley = sv_reg.compute_shapley(ind_feat=5)

def test_check_norm_shap(cache_norm_shap):
    test_dict = cache_norm_shap
    assert test_dict['r_squared_full'] == test_dict['sum_shaps']

def test_data_sv():
    dataset = "data/base_test_sv_reg_working.csv"
    sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])
    data = sv_reg.data_sv
    assert data.shape == (643, 28)

def test_fit():
    dataset = "data/base_test_sv_reg_working.csv"
    sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])
    sv_reg.fit()
    assert sv_reg.x_features.shape == (643, 5)

def test_fit_incorrect():
    dataset = "data/base_test_sv_reg_working.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(data=dataset, target="incorrect_t", ind_predictors_selected=[0, 1, 2, 3, 4, 5])
        sv_reg.fit()

def test_fit_incorrect_2():
    dataset = "data/base_test_sv_reg_working.csv"
    with pytest.raises(ValueError):
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[])
        sv_reg.fit()