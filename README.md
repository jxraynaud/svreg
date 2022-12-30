### svReg

An utility to perform Regression using Shapley Values from Game Theory.

----

## Installation

```bash
pip install -i https://test.pypi.org/simple/ sv_regression
```

| Function  | Description  |
| :------------- |:-----------------------------------------------------------------------------------|
| list_r_squared | Contains the R^2 of regressions computed from the differents coalitions of features
| normalize      | Normalize features and targets selected
| unnormalize    | Denormalize features and targets selected
| compute_usefullness | Compute the usefulness corresponding to the coalition of predictors "coalition" with the target predictor "target"
| compute_shapley | Compute shapley value using target_pred as the indice of the predictor of interest
| fit | Compute the coefficients of the regression using the shapley values
| check_norm_shap | Compute both R^2 of the full model and the sum of the shapley values.


### Requirements

- Python 3.9 or higher

---

## Usage

```python
from sv_regression import SvRegression
```

### Changelog

- 0.1.0: Initial release
