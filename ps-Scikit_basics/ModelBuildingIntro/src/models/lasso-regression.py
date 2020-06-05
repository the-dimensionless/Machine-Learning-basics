import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

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

from sklearn.linear_model import Lasso

# alpha -> Hyperparameter
lasso_model = Lasso(
    alpha=0.5, normalize=True
)  # normlize (subtract mean and divide by L2 norm)
lasso_model.fit(X_train, Y_train)

predictors = X_train.columns
r_score = lasso_model.score(X_train, Y_train)
print("lasso score ", r_score)

coef = pd.Series(lasso_model.coef_, predictors).sort_values()
# print("coef of lasso\n", coef)

y_predict = lasso_model.predict(x_test)

""" # Plot the graph
plot.plot(y_predict, label="Predicted")
plot.plot(y_test.values, label="Actual")
plot.ylabel("Price")
plot.legend()
plot.show()
 """
r_square = lasso_model.score(x_test, y_test)
print("residual square", r_square)

from sklearn.metrics import mean_squared_error
import math as m

lasso_me = mean_squared_error(y_predict, y_test)
lasso_rme = m.sqrt(lasso_me)

print("lasso rms ", lasso_rme)

# tweak alpha and check it
