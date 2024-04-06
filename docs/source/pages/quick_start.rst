============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/deforce />`_::

   $ pip install deforce==1.0.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/deforce.git
   $ cd deforce
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/deforce


After installation, you can import deforce as any other Python module::

   $ python
   >>> import deforce
   >>> deforce.__version__

========
Examples
========

Please check all use cases and examples in folder `examples <https://github.com/thieu1995/deforce/tree/main/examples />`_.


1) deforce provides several useful classes
------------------------------------------

All classes ::

   from deforce import DataTransformer, Data
   from deforce import CfnRegressor, CfnClassifier
   from deforce import DfoCfnRegressor, DfoCfnClassifier


2) What you can do with `DataTransformer` class
-----------------------------------------------

We provide many scaler classes that you can select and make a combination of transforming your data via
DataTransformer class. For example:

2.1) I want to scale data by `Loge` and then `Sqrt` and then `MinMax`:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example code::

	from deforce import DataTransformer
	import pandas as pd
	from sklearn.model_selection import train_test_split

	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:5].values
	y = dataset.iloc[:, 5].values
	X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

	dt = DataTransformer(scaling_methods=("loge", "sqrt", "minmax"))
	X_train_scaled = dt.fit_transform(X_train)
	X_test_scaled = dt.transform(X_test)


2.2) I want to scale data by `YeoJohnson` and then `Standard`:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example code::

	from deforce import DataTransformer
	import pandas as pd
	from sklearn.model_selection import train_test_split

	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:5].values
	y = dataset.iloc[:, 5].values
	X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

	dt = DataTransformer(scaling_methods=("yeo-johnson", "standard"))
	X_train_scaled = dt.fit_transform(X_train)
	X_test_scaled = dt.transform(X_test)


3) What can you do with `Data` class
------------------------------------

+ You can load your dataset into Data class
+ You can split dataset to train and test set
+ You can scale dataset without using DataTransformer class
+ You can scale labels using LabelEncoder

Example code::

	from deforce import Data
	import pandas as pd

	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:5].values
	y = dataset.iloc[:, 5].values

	data = Data(X, y, name="position_salaries")

	#### Split dataset into train and test set
	data.split_train_test(test_size=0.2, shuffle=True, random_state=100, inplace=True)

	#### Feature Scaling
	data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "sqrt", "minmax"))
	data.X_test = scaler_X.transform(data.X_test)

	data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
	data.y_test = scaler_y.transform(data.y_test)


4) What can you do with all model classes
-----------------------------------------

+ Define the model
+ Use provides functions to train, predict, and evaluate model

Example code::

	from deforce import CfnRegressor, CfnClassifier, DfoCfnRegressor, DfoCfnClassifier

	## Use standard CFN model for regression problem
	regressor = CfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
		max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)

	## Use standard CFN model for classification problem
	classifier = CfnClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
		max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)

	## Use Metaheuristic-optimized CFN model for regression problem
	print(DfoCfnClassifier.SUPPORTED_OPTIMIZERS)
	print(DfoCfnClassifier.SUPPORTED_REG_OBJECTIVES)

	opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
	regressor = DfoCfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
		obj_name="MSE", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)

	## Use Metaheuristic-optimized CFN model for classification problem
	print(DfoCfnClassifier.SUPPORTED_OPTIMIZERS)
	print(DfoCfnClassifier.SUPPORTED_CLS_OBJECTIVES)

	opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
	classifier = DfoCfnClassifier(hidden_size=50, act1_name="tanh", act2_name="softmax",
		obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)


5) What can you do with model object
------------------------------------

Example code::

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
	model.save_evaluation_metrics(data.y_test, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv")

	## Save training loss to csv file
	model.save_training_loss(save_path="history", filename="loss.csv")

	## Save predicted label
	model.save_y_predicted(X=data.X_test, y_true=data.y_test, save_path="history", filename="y_predicted.csv")

	## Save model
	model.save_model(save_path="history", filename="traditional_CFN.pkl")

	## Load model
	trained_model = CfnRegressor.load_model(load_path="history", filename="traditional_CFN.pkl")


In this section, we will explore the usage of the deforce model with the assistance of a dataset. While all the
preprocessing steps mentioned below can be replicated using Scikit-Learn, we have implemented some utility functions
to provide users with convenience and faster usage.


6) Combine deforce library like a normal library with scikit-learn:
-------------------------------------------------------------------

Example code::

	### Step 1: Importing the libraries
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import MinMaxScaler, LabelEncoder
	from deforce import CfnRegressor, CfnClassifier, DfoCfnRegressor, DfoCfnClassifier

	#### Step 2: Reading the dataset
	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:2].values
	y = dataset.iloc[:, 2].values

	#### Step 3: Next, split dataset into train and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=100)

	#### Step 4: Feature Scaling
	scaler_X = MinMaxScaler()
	scaler_X.fit(X_train)
	X_train = scaler_X.transform(X_train)
	X_test = scaler_X.transform(X_test)

	le_y = LabelEncoder()  # This is for classification problem only
	le_y.fit(y)
	y_train = le_y.transform(y_train)
	y_test = le_y.transform(y_test)

	#### Step 5: Fitting MLP-based model to the dataset

	##### 5.1: Use standard MLP model for regression problem
	regressor = CfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
		max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)
	regressor.fit(X_train, y_train)

	##### 5.2: Use standard MLP model for classification problem
	classifer = CfnClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
		max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)
	classifer.fit(X_train, y_train)

	##### 5.3: Use Metaheuristic-based MLP model for regression problem
	print(DfoCfnClassifier.SUPPORTED_OPTIMIZERS)
	print(DfoCfnClassifier.SUPPORTED_REG_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	model = DfoCfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
		obj_name="MSE", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
	regressor.fit(X_train, y_train)

	##### 5.4: Use Metaheuristic-based MLP model for classification problem
	print(DfoCfnClassifier.SUPPORTED_OPTIMIZERS)
	print(DfoCfnClassifier.SUPPORTED_CLS_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	classifier = DfoCfnClassifier(hidden_size=50, act1_name="tanh", act2_name="softmax",
		obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
	classifier.fit(X_train, y_train)

	#### Step 6: Predicting a new result
	y_pred = regressor.predict(X_test)

	y_pred_cls = classifier.predict(X_test)
	y_pred_label = le_y.inverse_transform(y_pred_cls)

	#### Step 7: Calculate metrics using score or scores functions.
	print("Try my AS metric with score function")
	print(regressor.score(X_test, y_test, method="AS"))

	print("Try my multiple metrics with scores function")
	print(classifier.scores(X_test, y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))


7) Utilities everything that deforce provided:
----------------------------------------------

Example code::

	### Step 1: Importing the libraries
	from deforce import Data, CfnRegressor, CfnClassifier, DfoCfnRegressor, DfoCfnClassifier
	from sklearn.datasets import load_digits

	#### Step 2: Reading the dataset
	X, y = load_digits(return_X_y=True)
	data = Data(X, y)

	#### Step 3: Next, split dataset into train and test set
	data.split_train_test(test_size=0.2, shuffle=True, random_state=100)

	#### Step 4: Feature Scaling
	data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("minmax"))
	data.X_test = scaler_X.transform(data.X_test)

	data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
	data.y_test = scaler_y.transform(data.y_test)

	#### Step 5: Fitting MLP-based model to the dataset
	##### 5.1: Use standard MLP model for regression problem
	regressor = CfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
		max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)
	regressor.fit(data.X_train, data.y_train)

	##### 5.2: Use standard MLP model for classification problem
	classifer = CfnClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
		max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)
	classifer.fit(data.X_train, data.y_train)

	##### 5.3: Use Metaheuristic-based MLP model for regression problem
	print(DfoCfnClassifier.SUPPORTED_OPTIMIZERS)
	print(DfoCfnClassifier.SUPPORTED_REG_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	model = DfoCfnRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
		obj_name="MSE", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
	regressor.fit(data.X_train, data.y_train)

	##### 5.4: Use Metaheuristic-based MLP model for classification problem
	print(DfoCfnClassifier.SUPPORTED_OPTIMIZERS)
	print(DfoCfnClassifier.SUPPORTED_CLS_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	classifier = DfoCfnClassifier(hidden_size=50, act1_name="tanh", act2_name="softmax",
		obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
	classifier.fit(data.X_train, data.y_train)

	#### Step 6: Predicting a new result
	y_pred = regressor.predict(data.X_test)

	y_pred_cls = classifier.predict(data.X_test)
	y_pred_label = scaler_y.inverse_transform(y_pred_cls)

	#### Step 7: Calculate metrics using score or scores functions.
	print("Try my AS metric with score function")
	print(regressor.score(data.X_test, data.y_test, method="AS"))

	print("Try my multiple metrics with scores function")
	print(classifier.scores(data.X_test, data.y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))


A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize
the data within a particular range.

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
