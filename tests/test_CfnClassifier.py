#!/usr/bin/env python
# Created by "Thieu" at 11:27, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from deforce import CfnClassifier


def test_CfnClassifier_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    model = CfnClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
                          max_epochs=100, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False, seed=42)
    model.fit(X, y)
    pred = model.predict(X)
    assert CfnClassifier.SUPPORTED_CLS_METRICS == model.SUPPORTED_CLS_METRICS
    assert pred[0] in (0, 1)
