#!/usr/bin/env python
# Created by "Thieu" at 21:45, 06/04/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_diabetes
from mealpy import StringVar, IntegerVar, MixedSetVar
from deforce import Data, DfoTuneCfn


## Load data object
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
data.y_test = scaler_y.transform(data.y_test.reshape(-1, 1))

# Design the boundary (parameters)

# class CfnRegressor
#     def __init__(self, hidden_size=50, act1_name="tanh", act2_name="sigmoid",
#                  obj_name="MSE", max_epochs=1000, batch_size=32, optimizer="SGD",
#                  optimizer_paras=None, verbose=False, seed=None, **kwargs):

my_bounds = [
    IntegerVar(lb=5, ub=21, name="hidden_size"),
    StringVar(valid_sets=("relu", "leaky_relu", "celu", "prelu", "gelu",
                          "elu", "selu", "rrelu", "tanh", "sigmoid"), name="act1_name"),
    StringVar(valid_sets=("relu", "leaky_relu", "celu", "prelu", "gelu",
                          "elu", "selu", "rrelu", "tanh", "sigmoid"), name="act2_name"),
    IntegerVar(lb=700, ub=1000, name="max_epochs"),
    MixedSetVar(valid_sets=((8, 16, 32, 64)), name="batch_size"),
    StringVar(valid_sets=("Adadelta", "Adagrad", "Adam", "Adamax", "AdamW", "ASGD", "LBFGS",
                          "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"), name="optimizer"),
]

opt_paras = {"name": "WOA", "epoch": 10, "pop_size": 20}
model = DfoTuneCfn(problem_type="regression", bounds=my_bounds, cv=3, scoring="MSE",
                      optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)
model.fit(data.X_train, data.y_train)
print(model.best_params)
print(model.best_estimator)
print(model.best_estimator.scores(data.X_test, data.y_test, list_methods=("MSE", "RMSE", "MAPE", "NSE", "R2", "KGE")))
