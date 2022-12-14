from sv_regression_api import SvRegression, LinearRegression, StandardScaler
from unittest import TestCase

class TryTesting(TestCase):
    def test_init(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        self.assertEqual(sv_reg.x_features.shape, (643, 27))

    def test_init_2(self):
        dataset = "data/base_test_sv_reg_working.csv"
        try:
            sv_reg = SvRegression(data=dataset, target="invalid_target")
            self.assertTrue(False)
        except:
            self.assertTrue(True)  

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
        x_features_norm, y_target_norm = sv_reg.normalize()
        dum_num = 5
        dum_predictors = list(range(dum_num))
        x_features_norm_dum = x_features_norm[:, dum_predictors]
        lin_reg = LinearRegression(n_jobs=-1, copy_X=False)
        rsquared = sv_reg._get_rsquared_sk(sv_reg.num_feat_selec)
        self.assertEqual(rsquared, 0.8848637227130571) 
        
    def test_get_rsquared_2(self): 
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        x_features_norm, y_target_norm = sv_reg.normalize()
        dum_num = 5
        dum_predictors = list(range(dum_num))
        x_features_norm_dum = x_features_norm[:, dum_predictors]
        lin_reg = LinearRegression(n_jobs=-1, copy_X=False)
        rsquared = sv_reg._get_rsquared_sk(0)
        self.assertEqual(rsquared, 0)


    def test_fit_5_features(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])
        values = [0.04115089, 0.77510499, 0.21830721, 0.08091167, 0.96657913]
        fit = sv_reg.fit()
        for i in range(len(values)):
            if round (fit[i], 3) != round(values[i], 3):
                self.assertEqual(round (fit[i], 3), round(values[i], 3))    
        self.assertTrue(True)

    def test_fit_10_features(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        values = [0.11084141, 0.16932976, 0.03767706, 0.08262286, 0.07923011, 0.05066642, 0.39426036, 0.18698367, 0.11121645, 0.3680613]
        fit = sv_reg.fit()
        for i in range(len(values)):
            if round (fit[i], 3) != round(values[i], 3):
                self.assertEqual(round (fit[i], 3), round(values[i], 3))    
        self.assertTrue(True)   

    def test_compute_shapley_incorrect_target(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0, 1, 2, 3, 4])   
        try:
            shapley = sv_reg.compute_shapley(target_pred=6)
            self.assertTrue(False)
        except:
            self.assertTrue(True)   


    def test_compute_shapley_1_feature(self):
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto", ind_predictors_selected=[0])
        shapley = sv_reg.compute_shapley(target_pred=0)
        self.assertEqual(round(shapley, 3), 0.733)  

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
