# Gradient Boosting Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.ensemble import GradientBoostingRegressor

auto_data = pd.read_csv("../data/imports-85.data", sep=r"\s*,\s*", engine="python")
auto_data = auto_data.replace("?", np.nan)  # make it NaN

auto_data["price"] = pd.to_numeric(auto_data["price"], errors="coerce")
auto_data["horsepower"] = pd.to_numeric(auto_data["horsepower"], errors="coerce")

auto_data = auto_data.drop("normalized-losses", axis=1)

cylinders_dict = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "eight": 8,
    "twelve": 12,
}

auto_data["num-of-cylinders"].replace(cylinders_dict, inplace=True)

# change categorical data to one hot representation

auto_data = pd.get_dummies(
    auto_data,
    columns=[
        "make",
        "fuel-type",
        "aspiration",
        "body-style",
        "drive-wheels",
        "engine-location",
        "engine-type",
        "fuel-system",
        "num-of-doors",
    ],
)

auto_data = auto_data.dropna()  # drop rows with NaN
# check for null values
# print(auto_data[auto_data.isnull().any(axis=1)])

# train test spli

from sklearn.model_selection import train_test_split

X = auto_data.drop("price", axis=1)

# taking the labels (price)
Y = auto_data["price"]

# Split 80:20
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


params = {
    "n_estimators": 500,  # boosting stages
    "max_depth": 6,
    "min_samples_split": 2,
    "learning_rate": 0.01,
    "loss": "ls",  # least square
}

gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, Y_train)

print("score : ", gbr_model.score(X_train, Y_train))

y_predicted = gbr_model.predict(x_test)

""" 
plot.plot(y_predicted, label="Predicted")
plot.plot(y_test.values, label="Actual")
plot.ylabel("MPG")

plot.legend()
plot.show()
 """

r_square = gbr_model.score(x_test, y_test)
print("r square: ", r_square)

from sklearn.metrics import mean_squared_error
import math

gbr_mse = mean_squared_error(y_predicted, y_test)
print("mse: ", gbr_mse)

gbr_rmse = math.sqrt(gbr_mse)
print("rsme: ", gbr_rmse)

print("--------------------------------------------")

from sklearn.model_selection import GridSearchCV

num_estimators = [100, 200, 500]
learn_rates = [0.01, 0.02, 0.05, 0.1]
max_depths = [4, 6, 8]

param_grid = {
    "n_estimators": num_estimators,
    "learning_rate": learn_rates,
    "max_depth": max_depths,
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(min_samples_split=2, loss="ls"),
    param_grid,
    cv=3,
    return_train_score=True,
)

grid_search.fit(X_train, Y_train)
print("best params : ", grid_search.best_params_)

# grid_search.cv_results_ # get all details
