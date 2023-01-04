import pandas as pd
import pytest

from svreg.svreg import SvRegression


@pytest.fixture
def dataset():
    dataset = "data/mtcars.csv"
    df_dataset = pd.read_csv(dataset, index_col="model")
    return df_dataset


@pytest.fixture
def cache_compute_1_feature(request, dataset):
    shapley = request.config.cache.get("shapley_1", None)
    if shapley is None:
        sv_reg = SvRegression(data=dataset, target="mpg", ind_predictors_selected=[0])
        shapley = sv_reg.compute_shapley()
        request.config.cache.set("shapley_1", shapley)
    return shapley


@pytest.fixture
def cache_compute_5_features(request, dataset):
    shapley = request.config.cache.get("shapley_2", None)
    if shapley is None:
        sv_reg = SvRegression(data=dataset, target="mpg", ind_predictors_selected=[0, 1, 2, 3, 4])
        shapley = sv_reg.compute_shapley()
        request.config.cache.set("shapley_2", shapley)
    return shapley


@pytest.fixture
def cache_norm_shap(request, dataset):
    test_dict = request.config.cache.get("test_dict", None)
    if test_dict is None:
        sv_reg = SvRegression(data=dataset, target="mpg", ind_predictors_selected=[0, 1, 2, 3, 4])
        test_dict = sv_reg.check_norm_shap()
        request.config.cache.set("test_dict", test_dict)
    return test_dict
