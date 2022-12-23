import pytest
from sv_regression.sv_regression import SvRegression

@pytest.fixture
def cache_compute_5_features(request):
    dataset = "data/base_test_sv_reg_working.csv"
    shapley = request.config.cache.get("shapley_2", None)
    if shapley is None:
        sv_reg = SvRegression(
            data=dataset,
            target="qlead_auto",
            ind_predictors_selected=[0, 1, 2, 3, 4],
        )
        shapley = sv_reg.compute_shapley()
        request.config.cache.set("shapley_2", shapley)
    return shapley


@pytest.fixture
def cache_norm_shap(request):
    dataset = "data/base_test_sv_reg_working.csv"
    test_dict = request.config.cache.get("test_dict", None)
    if test_dict is None:
        sv_reg = SvRegression(
            data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4]
        )
        test_dict = sv_reg.check_norm_shap()
        request.config.cache.set("test_dict", test_dict)
    return test_dict

