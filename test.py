from sv_regression import *
import style
def test_0_1():
    try:
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        return (0)
    except Exception as e:
        print(style.red('Error: ', e))
        return (1)

def test_0_2():
    try:
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        x_features_norm, y_target_norm = sv_reg.normalize()
        return (0)
    except Exception as e:
        print(style.red('Error: ', e))
        return (1)

def test_0_3():
    try:
        dataset = "data/base_test_sv_reg_working.csv"
        sv_reg = SvRegression(data=dataset, target="qlead_auto")
        x_features_norm, y_target_norm = sv_reg.normalize()
        x_features_un, y_target_un = sv_reg.unnormalize(x_features_norm, y_target_norm)
        return (0)
    except Exception as e:
        print(style.red('Error: ', e))
        return (1)

def test_1_0():
    try:
        dataset = "incorrect_path.csv"
        sv_reg = SvRegression(data=dataset, target="")
        x_features_norm, y_target_norm = sv_reg.normalize()
        return (1)
    except Exception as e:
        print(style.green('Error: ', e))
        return (0)


def test_0():
    print('test_0_1: ', '✅' if test_0_1() == 0 else '❌')
    print('test_0_2 (normalize): ', '✅' if test_0_2() == 0 else '❌')
    print('test_0_3 (unnormalize): ', '✅' if test_0_3() == 0 else '❌')
    print('test_0_3 (unnormalize): ', '✅' if test_1_0() == 0 else '❌')


if __name__ == '__main__':

    # test_0()



    dataset = "data/base_test_sv_reg_working.csv"
    sv_reg = SvRegression(data=dataset,
                        target="qlead_auto")

    x_features_norm, y_target_norm = sv_reg.normalize()

    dum_num = 5
    dum_predictors = list(range(dum_num))
    x_features_norm_dum = x_features_norm[:, dum_predictors]

    # declaring model.
    # lin_reg = LinearRegression(n_jobs=-1, copy_X=False)
    # def get_rsquared_sk(ind=None):
    #     print('ind value: ', ind)
    #     if ind == 0:
    #         return 0.0
    #     else:
    # #         # TODO: find a way to format the format code dynamically (e.g replace '5' by dum_num at runtime).
    #         ind_bin = f"{ind:05b}"
    #         print('ind_bin: ', ind_bin)
    #         dum_mask = [bool(int(ind_f)) for ind_f in ind_bin[::-1]]
    #         x_features_curr = x_features_norm_dum[:, dum_mask]
    #         lin_reg.fit(x_features_curr, y_target_norm)
    #         return lin_reg.score(x_features_curr, y_target_norm)

    # # Activate GPU acceleration.
    # # Problem: requires dpctl to work:
    # # https://pypi.org/project/dpctl/

    # # I had an issue installing dpctl, turning off for now.
    # # with config_context(target_offload="gpu:0"):

    # r_squared_dum_compr is defined as a global variable to avoid memory overload.
    start = timer()
    r_squared_dum_compr = [get_rsquared_sk(ind) for ind in range(0, 2**dum_num)]
    print('>>>>>' * 6)
    print(r_squared_dum_compr)
    print('>>>>>' * 6)
    time_comp = timer() - start

    # # Comptue usefullness as defined by formulae 18 from the article.
    # # We do not pass r_squared_dum_compr (set it globally) as a parameter to avoid memory overload.
    # def compute_usefullness(predictors=None, target=2, len_predictors=4):

    #     if len(predictors) == 1:
    #         # Rsquared corresponding to length 1 predictors:
    #         bin_predictors = [1 if x in predictors else 0 for x in range(len_predictors)]
    #         ind_rsquared = int("".join(str(x) for x in bin_predictors), 2)

    #         r_squared = r_squared_dum_compr[ind_rsquared]

    #         return r_squared

    #     else:
    #         # Rsquared with target:
    #         bin_predictors = [1 if x in predictors else 0 for x in range(len_predictors)]
    #         ind_rsquared = int("".join(str(x) for x in bin_predictors), 2)
    #         r_squared_with_target = r_squared_dum_compr[ind_rsquared]

    #         # Rsquared without target:
    #         # predictors.remove(target)  # bad idea to use list.remove() --> add side effects to the function.
    #         predictors = [x for x in predictors if x is not target]
    #         bin_predictors_without_target = [1 if x in predictors else 0 for x in range(len_predictors)]
    #         ind_rsquared_without_target = int("".join(str(x) for x in bin_predictors_without_target), 2)
    #         r_squared_with_target_without_target = r_squared_dum_compr[ind_rsquared_without_target]

    #         return r_squared_with_target - r_squared_with_target_without_target


    # def compute_shapley(target_pred=None, predictors=None):
    #     if target_pred not in predictors:
    #         raise ValueError(f"""\npredictors: \n{predictors}.\ntarget_pred:\n{target_pred}\n""" +
    #                         """target_pred must be in predictors.""")
    #     num_predictors = len(predictors)
    #     shapley_val = 0
    # # First, second, third etc... term
    #     for len_comb in range(num_predictors):
    #         sum_usefullness = 0
    #         weight = (np.math.factorial(len_comb) * np.math.factorial(num_predictors - len_comb - 1)) / np.math.factorial(num_predictors)
    #         # Checking that the weigths are correct (see eq. 17).
    #         # ic(len_comb)
    #         # ic(weight)
    #         # input("waiting\n")
    #         # The weights are correct...
    #         for coalition in filter(lambda x: target_pred in x, combinations(predictors, len_comb)):
    #             usefullness = compute_usefullness(predictors=coalition,
    #                                             target=target_pred,
    #                                             len_predictors=len(predictors))
    #             sum_usefullness = sum_usefullness + usefullness
    #         shapley_val = shapley_val + weight * sum_usefullness

    #     return shapley_val

    # # Testing that the sum of Shapley values is equal to the complete r_squared.

    # # Just to make sure we are using the correct predictors

    #     x_features_norm_dum_test = x_features_norm[:, dum_predictors]

    #     r_squared_full = lin_reg.fit(x_features_norm_dum_test, y_target_norm).score(x_features_norm_dum_test, y_target_norm)

    #     sum_shap = 0.0

    #     for ind_feat in dum_predictors:

    #         shap = compute_shapley(target_pred=ind_feat, predictors=dum_predictors)
    #         sum_shap = sum_shap + shap

    #     ic(r_squared_full)
    #     ic(sum_shap)