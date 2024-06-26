#!/usr/bin/env python
# Created by "Thieu" at 17:43, 13/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from typing import Tuple
import numpy as np
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from deforce.model.base_cfn_numpy import BaseDfoCfn, CfnNumpy
from deforce.toolkit.scalers import ObjectiveScaler


class DfoCfnRegressor(BaseDfoCfn, RegressorMixin):
    """
    Defines the general class of Metaheuristic-based CFN model for Regression problems that inherit the BaseDfoCfn and RegressorMixin classes.

    Parameters
    ----------
    hidden_size : int, default=50
        The number of hidden nodes

    act1_name : str, default='tanh'
        Activation function for the hidden layer. The supported activation functions are:
        ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "swish", "hard_swish", "soft_plus", "mish", "soft_sign",
        "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax", "silu"]

    act2_name : str, default='sigmoid'
        Activation function for the hidden layer. The supported activation functions are:
        ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "swish", "hard_swish", "soft_plus", "mish", "soft_sign",
        "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax", "silu"]

    obj_name : str, default="MSE"
        Current supported objective functions, please check it here: https://github.com/thieu1995/permetrics

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "OriginalWOA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    verbose : bool, default=True
        Whether to print progress messages to stdout.

    seed : int, default=None
        Seed value to reproduce the results.

    Examples
    --------
    >>> from deforce import DfoCfnRegressor, Data
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> data.X_train_scaled, scaler = data.scale(data.X_train, method="MinMaxScaler")
    >>> data.X_test_scaled = scaler.transform(data.X_test)
    >>> opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    >>> model = DfoCfnRegressor(hidden_size=15, act1_name="relu", act2_name="sigmoid",
    >>>             obj_name="MSE", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True, seed=42)
    >>> model.fit(data.X_train_scaled, data.y_train)
    >>> pred = model.predict(data.X_test_scaled)
    >>> print(pred)
    """

    def __init__(self, hidden_size=50, act1_name="tanh", act2_name="sigmoid",
                 obj_name="MSE", optimizer="OriginalWOA", optimizer_paras=None, verbose=True, seed=None):
        super().__init__(hidden_size=hidden_size, act1_name=act1_name, act2_name=act2_name,
            obj_name=obj_name, optimizer=optimizer, optimizer_paras=optimizer_paras, verbose=verbose, seed=seed)

    def create_network(self, X, y) -> Tuple[CfnNumpy, ObjectiveScaler]:
        """
        Returns
        -------
            network: CFN, an instance of CFN network
            obj_scaler: ObjectiveScaler, the objective scaler that used to scale output
        """
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                size_output = 1
            elif y.ndim == 2:
                size_output = y.shape[1]
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector or 2D matrix.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        if size_output > 1:
            if self.obj_weights is None:
                self.obj_weights = 1./size_output * np.ones(size_output)
            elif self.obj_weights in (list, tuple, np.ndarray):
                if not (len(self.obj_weights) == size_output):
                    raise ValueError(f"There is {size_output} objectives, but obj_weights has size of {len(self.obj_weights)}")
            else:
                raise TypeError("Invalid obj_weights array type, it should be list, tuple or np.ndarray")
        obj_scaler = ObjectiveScaler(obj_name="self", ohe_scaler=None)
        network = CfnNumpy(input_size=X.shape[1], hidden_size=self.hidden_size, output_size=size_output,
                           act1_name=self.act1_name, act2_name=self.act2_name)
        return network, obj_scaler

    def objective_function(self, solution=None):
        """
        Evaluates the fitness function for regression metric

        Parameters
        ----------
        solution : np.ndarray, default=None

        Returns
        -------
        result: float
            The fitness value
        """
        self.network.update_weights_from_solution(solution)
        y_pred = self.network.predict(self.X_temp)
        loss_train = RegressionMetric(self.y_temp, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return loss_train

    def score(self, X, y, method="RMSE"):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseDfoCfn__score_reg(X, y, method)

    def scores(self, X, y, list_methods=("MSE", "MAE")):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseDfoCfn__scores_reg(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("MSE", "MAE")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseDfoCfn__evaluate_reg(y_true, y_pred, list_metrics)


class DfoCfnClassifier(BaseDfoCfn, ClassifierMixin):
    """
    Defines the general class of Metaheuristic-based CFN model for Classification problems that inherit the BaseDfoCfn and ClassifierMixin classes.

    Parameters
    ----------
    hidden_size : int, default=50
        The number of hidden nodes

    act1_name : str, default='tanh'
        Activation function for the hidden layer. The supported activation functions are:
        ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "swish", "hard_swish", "soft_plus", "mish", "soft_sign",
        "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax", "silu"]

    act2_name : str, default='sigmoid'
        Activation function for the hidden layer. The supported activation functions are:
        ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "swish", "hard_swish", "soft_plus", "mish", "soft_sign",
        "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax", "silu"]

    obj_name : str, default="MSE"
        Current supported objective functions, please check it here: https://github.com/thieu1995/permetrics

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "OriginalWOA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    verbose : bool, default=True
        Whether to print progress messages to stdout.

    seed : int, default=None
        Seed value to reproduce the results.

    Examples
    --------
    >>> from deforce import Data, DfoCfnClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> data.X_train_scaled, scaler = data.scale(data.X_train, method="MinMaxScaler")
    >>> data.X_test_scaled = scaler.transform(data.X_test)
    >>> opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    >>> print(DfoCfnClassifier.SUPPORTED_CLS_OBJECTIVES)
    {'PS': 'max', 'NPV': 'max', 'RS': 'max', ...., 'KLDL': 'min', 'BSL': 'min'}
    >>> model = DfoCfnClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
    >>>             obj_name="NPV", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)
    >>> model.fit(data.X_train_scaled, data.y_train)
    >>> pred = model.predict(data.X_test_scaled)
    >>> print(pred)
    array([1, 0, 1, 0, 1])
    """

    CLS_OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, hidden_size=50, act1_name="tanh", act2_name="sigmoid",
                 obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=None, verbose=True, seed=None):
        super().__init__(hidden_size=hidden_size, act1_name=act1_name, act2_name=act2_name,
            obj_name=obj_name, optimizer=optimizer, optimizer_paras=optimizer_paras, verbose=verbose, seed=seed)
        self.return_prob = False

    def _check_y(self, y):
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                return len(np.unique(y))
            raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")

    def create_network(self, X, y) -> Tuple[CfnNumpy, ObjectiveScaler]:
        self.n_labels = self._check_y(y)
        if self.n_labels > 2:
            if self.obj_name in self.CLS_OBJ_LOSSES:
                self.return_prob = True
        ohe_scaler = OneHotEncoder(sparse_output=False)
        ohe_scaler.fit(np.reshape(y, (-1, 1)))
        obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
        network = CfnNumpy(input_size=X.shape[1], hidden_size=self.hidden_size, output_size=self.n_labels,
                           act1_name=self.act1_name, act2_name=self.act2_name)
        return network, obj_scaler

    def objective_function(self, solution=None):
        """
        Evaluates the fitness function for classification metric

        Parameters
        ----------
        solution : np.ndarray, default=None

        Returns
        -------
        result: float
            The fitness value
        """
        self.network.update_weights_from_solution(solution)
        y_pred = self.predict(self.X_temp, return_prob=self.return_prob)
        y1 = self.obj_scaler.inverse_transform(self.y_temp)
        loss_train = ClassificationMetric(y1, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return loss_train

    def score(self, X, y, method="AS"):
        """
        Return the metric on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        method : str, default="AS"
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseDfoCfn__score_cls(X, y, method)

    def scores(self, X, y, list_methods=("AS", "RS")):
        """
        Return the list of metrics on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        list_methods : list, default=("AS", "RS")
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseDfoCfn__scores_cls(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Return the list of performance metrics on the given test data and labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("AS", "RS")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseDfoCfn__evaluate_cls(y_true, y_pred, list_metrics)
