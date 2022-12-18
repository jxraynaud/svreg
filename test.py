from sv_regression import SvRegression, LinearRegression, StandardScaler
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
        with pytest.raises(ValueError):
            sv_reg = SvRegression(data=dataset, target="incorrect_t", ind_predictors_selected=[0, 1, 2, 3, 4])   
            shapley = sv_reg.compute_shapley(target_pred=6) 


    def test_compute_shapley_1_feature(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0])
        shapley = sv_reg.compute_shapley(ind_feat=0)
        self.assertEqual(round(shapley, 1), 0.7)  

    def test_compute_shapley_5_features(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])
        shapley = sv_reg.compute_shapley()
        self.assertEqual(round(shapley, 3), 0.128)

    def test_compute_shapley_10_features(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        shapley = sv_reg.compute_shapley()
        self.assertEqual(round(shapley, 3), 0.022)

    def test_compute_shapley_none(self):
        dataset = "data/base_test_sv_reg_working.csv"
        with pytest.raises(ValueError):
            sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[])
            shapley = sv_reg.compute_shapley()

    def test_compute_shapley_incorrect(self):
        dataset = "data/base_test_sv_reg_working.csv"
        with pytest.raises(ValueError):
            sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0])
            sv_reg._list_r_squared = None
            shapley = sv_reg.compute_shapley(ind_feat=5)    

    def test_check_norm_shap(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])
        # shapley = sv_reg.compute_shapley()
        test_dict = sv_reg.check_norm_shap()
        # compare the values of my dict
        self.assertEqual(test_dict['r_squared_full'], test_dict['sum_shaps'])  

    def test_data_sv(self):
        dataset = "data/base_test_sv_reg_working.csv"       
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])
        data = sv_reg.data_sv
        self.assertEqual(data.shape, (643, 28))

    def test_fit(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])
        sv_reg.fit()
        self.assertEqual(sv_reg.x_features.shape, (643, 5))

    def test_fit_incorrect(self):
        dataset = "data/base_test_sv_reg_working.csv"
        with pytest.raises(ValueError):
            sv_reg = SvRegression(data=dataset, target="incorrect_t", ind_predictors_selected=[0, 1, 2, 3, 4, 5])
            sv_reg.fit()

    def test_fit_incorrect_2(self):
        dataset = "data/base_test_sv_reg_working.csv"
        with pytest.raises(ValueError):
            sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[])
            sv_reg.fit()


if __name__ == '__main__':
    dataset = "data/base_test_sv_reg_working.csv"
    # sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])
    # shapley = sv_reg.compute_shapley()
    # data = sv_reg.data_sv
    # print(data.shape)