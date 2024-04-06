#!/usr/bin/env python
# Created by "Thieu" at 11:25, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from deforce import CfnRegressor


def test_CfnRegressor_class():
    X = np.random.uniform(low=0.0, high=1.0, size=(100, 5))
    noise = np.random.normal(loc=0.0, scale=0.1, size=(100, 5))
    y = 2 * X + 1 + noise

    model = CfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
                         max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False, seed=42)
    model.fit(X, y)

    pred = model.predict(X)
    assert CfnRegressor.SUPPORTED_REG_METRICS == model.SUPPORTED_REG_METRICS
    assert len(pred) == X.shape[0]
