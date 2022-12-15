from sv_regression_api import SvRegression, LinearRegression, StandardScaler
from unittest import TestCase
import pytest

class TryTesting(TestCase):
    def test_init(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        self.assertEqual(sv_reg.x_features.shape, (643, 27))

    def test_init_2(self):
        dataset = "data/base_test_sv_reg_working.csv"
        with pytest.raises(ValueError):
            sv_reg = SvRegression(data=dataset, target="invalid_target")

    def test_normalize(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        x_features_norm, y_target_norm = sv_reg.normalize()
        self.assertEqual(x_features_norm.shape, (643, 27)) 

    def test_unnormalize(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        x_features_norm, y_target_norm = sv_reg.normalize()
        x_features, y_target = sv_reg.unnormalize(x_features_norm, y_target_norm)
        self.assertEqual(x_features.shape, (643, 27))   

    def test_get_rsquared(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        rsquared = sv_reg._get_rsquared_sk(sv_reg.num_feat_selec)
        self.assertEqual(round(rsquared, 3), 0.885) 
        
    def test_get_rsquared_2(self): 
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        rsquared = sv_reg._get_rsquared_sk(0)
        self.assertEqual(rsquared, 0)

    def test_compute_shapley_incorrect_target(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="incorrect_t", ind_predictors_selected=[0, 1, 2, 3, 4])   
        with pytest.raises(ValueError):
            shapley = sv_reg.compute_shapley(target_pred=6)


    def test_compute_shapley_1_feature(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0])
        shapley = sv_reg.compute_shapley(target_pred=0)
        self.assertEqual(shapley, 0)  

    def test_compute_shapley_5_features(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])
        shapley = sv_reg.compute_shapley()
        self.assertEqual(round(shapley, 3), 0.09)

    def test_compute_shapley_10_features(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        shapley = sv_reg.compute_shapley()
        self.assertEqual(round(shapley, 3), 0.007)

if __name__ == '__main__':
    dataset = "data/base_test_sv_reg_working.csv"
    sv_reg = SvRegression(data=dataset, target="invalid_target")
