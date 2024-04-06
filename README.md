
## deforce: Derivative-Free Algorithms for Optimizing Cascade Forward Neural Networks

---

[![GitHub release](https://img.shields.io/badge/release-1.0.0-yellow.svg)](https://github.com/thieu1995/deforce/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/deforce) 
[![PyPI version](https://badge.fury.io/py/deforce.svg)](https://badge.fury.io/py/deforce)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deforce.svg)
![PyPI - Status](https://img.shields.io/pypi/status/deforce.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deforce.svg)
[![Downloads](https://pepy.tech/badge/deforce)](https://pepy.tech/project/deforce)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/deforce/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/deforce/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/deforce.svg)
[![Documentation Status](https://readthedocs.org/projects/deforce/badge/?version=latest)](https://deforce.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/deforce.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10935437.svg)](https://doi.org/10.5281/zenodo.10935437)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

deforce (DErivative Free Optimization foR Cascade forward nEural networks) is a Python library that implements variants and the traditional version of Cascade Forward Neural Networks. These include Derivative Free-optimized CFN models (such as GA, PSO, WOA, TLO, DE, ...) and Gradient Descent-optimized CFN models (such as SGD, Adam, Adelta, Adagrad, ...). It provides a comprehensive list of optimizers for training CFN models and is also compatible with the Scikit-Learn library. With deforce, 
you can perform searches and hyperparameter tuning using the features provided by the Scikit-Learn library.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: CfnRegressor, CfnClassifier, DfoCfnRegressor, DfoCfnClassifier, DfoTuneCfn
* **Total DFO-based CFN Regressor**: > 200 Models 
* **Total DFO-based CFN Classifier**: > 200 Models
* **Total GD-based CFN Regressor**: 12 Models
* **Total GD-based CFN Classifier**: 12 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://deforce.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, torch, skorch


# Citation Request 

If you want to understand how to use Derivative Free-optimized Cascade Forward Neural Network, you 
need to read the paper titled **"Optimization of neural-network model using a meta-heuristic algorithm for the estimation of dynamic Poisson’s ratio of selected rock types"**. 
The paper can be accessed at the following [link](https://doi.org/10.1038%2Fs41598-023-38163-0)

Please include these citations if you plan to use this library:

```bibtex
@software{thieu_deforce_2024,
  author = {Van Thieu, Nguyen},
  title = {{deforce: Derivative-Free Algorithms for Optimizing Cascade Forward Neural Networks}},
  url = {https://github.com/thieu1995/deforce},
  doi = {10.5281/zenodo.10935437},
  year = {2024}
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}

@article{van2023groundwater,
  title={Groundwater level modeling using Augmented Artificial Ecosystem Optimization},
  author={Van Thieu, Nguyen and Barma, Surajit Deb and Van Lam, To and Kisi, Ozgur and Mahesha, Amai},
  journal={Journal of Hydrology},
  volume={617},
  pages={129034},
  year={2023},
  publisher={Elsevier}
}
```

# Installation

* Install the [current PyPI release](https://pypi.python.org/pypi/deforce):
```sh 
$ pip install deforce
```

After installation, check the installed version by:

```sh
$ python
>>> import deforce
>>> deforce.__version__
```

### Examples

Please check [documentation website](https://deforce.readthedocs.io/) and [examples folder](examples).

1) `deforce` provides this useful classes

```python
from deforce import DataTransformer, Data
from deforce import CfnRegressor, CfnClassifier
from deforce import DfoCfnRegressor, DfoCfnClassifier
```

2) What can you do with all model classes

```python
from deforce import CfnRegressor, CfnClassifier, DfoCfnRegressor, DfoCfnClassifier

## Use standard CFN model for regression problem
regressor = CfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
                         max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False, seed=42)

## Use standard CFN model for classification problem 
classifier = CfnClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
                           max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False, seed=42)

## Use Metaheuristic-optimized CFN model for regression problem
print(DfoCfnClassifier.SUPPORTED_OPTIMIZERS)
print(DfoCfnClassifier.SUPPORTED_REG_OBJECTIVES)

opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
regressor = DfoCfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
                            obj_name="MSE", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)

## Use Metaheuristic-optimized CFN model for classification problem
print(DfoCfnClassifier.SUPPORTED_OPTIMIZERS)
print(DfoCfnClassifier.SUPPORTED_CLS_OBJECTIVES)

opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
classifier = DfoCfnClassifier(hidden_size=50, act1_name="tanh", act2_name="softmax",
                              obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)
```

3) After you define the model, do something with it
+ Use provides functions to train, predict, and evaluate model

```python
from deforce import CfnRegressor, Data

data = Data()  # Assumption that you have provide this object like above

model = CfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
                     max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)

## Train the model
model.fit(data.X_train, data.y_train)

## Predicting a new result
y_pred = model.predict(data.X_test)

## Calculate metrics using score or scores functions.
print(model.score(data.X_test, data.y_test, method="MAE"))
print(model.scores(data.X_test, data.y_test, list_methods=["MAPE", "NNSE", "KGE", "MASE", "R2", "R", "R2S"]))

## Calculate metrics using evaluate function
print(model.evaluate(data.y_test, y_pred, list_metrics=("MSE", "RMSE", "MAPE", "NSE")))

## Save performance metrics to csv file
model.save_evaluation_metrics(data.y_test, y_pred, list_metrics=("RMSE", "MAE"), save_path="history",
                              filename="metrics.csv")

## Save training loss to csv file
model.save_training_loss(save_path="history", filename="loss.csv")

## Save predicted label
model.save_y_predicted(X=data.X_test, y_true=data.y_test, save_path="history", filename="y_predicted.csv")

## Save model
model.save_model(save_path="history", filename="traditional_CFN.pkl")

## Load model 
trained_model = CfnRegressor.load_model(load_path="history", filename="traditional_CFN.pkl")
```
