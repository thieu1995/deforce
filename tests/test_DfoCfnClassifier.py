#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from deforce import DfoCfnClassifier


def test_DfoCfnClassifier_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = DfoCfnClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
                             obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)
    model.fit(X, y)
    pred = model.predict(X)
    assert DfoCfnClassifier.SUPPORTED_CLS_OBJECTIVES == model.SUPPORTED_CLS_OBJECTIVES
    assert pred[0] in (0, 1)
