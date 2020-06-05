# for classification problems
# same as SVM but with different objective function
import pandas as pd
import numpy as np
import matplotlib.pyplot as p
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

auto_data = pd.read_csv(
    "../data/auto-mpg.data",
    delim_whitespace=True,
    header=None,
    names=[
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model",
        "origin",
        "car_name",
    ],
)

auto_data = auto_data.drop("car_name", axis=1)

auto_data["origin"] = auto_data["origin"].replace(
    {1: "america", 2: "europe", 3: "asia"}
)

auto_data = pd.get_dummies(auto_data, columns=["origin"])
auto_data = auto_data.replace("?", np.nan)
auto_data - auto_data.dropna()
# print(auto_data)

# Regression pre
X = auto_data.drop("mpg", axis=1)  # all cols except mpg

# take labels
Y = auto_data["mpg"]  # only mpg col

# split 80:20

X_train, x_test, Y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_states=0
)

from sklearn.svm import SVR

# C -> Hyperparameter (penalty factor)
# epsilon -> another hyperparameter
model = SVR(kernel="linear", C=1.0)
model.fit(X_train, Y_train)

model_score = model.score(X_train, Y_train)
model_coef = model.coef_

predictors = X_train.columns
coef = pd.Series(model_coef[0], predictors).sort_values()
coef.plot(kind="bar", title="Model Coefficients")
