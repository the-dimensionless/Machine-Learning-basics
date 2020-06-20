import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
from pandas.plotting._matplotlib import scatter_matrix
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings

# wisdom comes from WHY questions and Skills from HOW

filepath = os.path.join(os.getcwd(), 'solution\\src\\dataset\\forestfires.csv')
cols = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC',
        'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']

# loading the data and defining the columns beforehand
df = pd.read_csv(filepath, names=cols)

""" # check if is null
print(pd.isnull(df))
 """

# Exploratory data analysis - numerical
print("===========================DATA SHAPE========================")
print(df.shape)

print("===========================DATA TYPES========================")
print(df.dtypes)
""" 
print("=====================Inspecting the head of data========================")
print(df.head(1))
 """
print()
# replace non numeric data to numeric ones
df.month.replace(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                  'aug', 'sep', 'oct', 'nov', 'dec'), ('1', '2', '3', '4', '5', '6', '7',
                                                       '8', '9', '10', '11', '12'), inplace=True)
df.day.replace(('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'), ('1', '2', '3', '4', '5', '6', '7'
                                                                   ), inplace=True)
print(df.head(5))

print("=====================Data Stats======================")
print(df.describe())

print("======================Correlations===================")
# Linear correlations (Small values are not reliable)
print(df.corr(method="pearson"))


# Exploratory data analysis - graphical
""" 
# histograms
df.hist(sharex=False, sharey=False, xlabelsize=3, ylabelsize=3)
plt.suptitle("Histograms", y=1.00, fontweight='bold')
plt.show()

# density
df.plot(kind='density', subplots=True, layout=(4, 4), sharex=False,
        fontsize=8)
plt.suptitle("Density", y=1.00, fontweight='bold')
plt.show()

# box and whisker plots
df.plot(kind='box', subplots=False, layout=(4, 4), sharex=False, sharey=False,
        fontsize=12)
plt.suptitle("Box and Whisker", y=1.00, fontweight='bold')
plt.show()


# region Bivariant
# scatter plot
scatter_matrix(df)
plt.suptitle("Scatter Matrix", y=1.00, fontweight='bold')
plt.show()

# correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0, 13, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(cols)
ax.set_yticklabels(cols)
plt.suptitle("Correlation Matrix", y=1.00, fontweight='bold')
plt.show()
 """

# K Fold Cross validation
""" 
dataset = range(16)
KFCrossValidator = KFold(n_splits=4, shuffle=False)
KFdataset = KFCrossValidator.split(dataset)
print('{} {:^61} {}'.format('Round', 'Training set', 'Testing set'))
for iteration, data in enumerate(KFdataset, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))
 """

# Regression
print('\nRegressions\n')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

array = df.values
X = array[:, 0:12]
Y = array[:, 12]

# Metrics
k_folds = 10
seed = 7
scoring = 'max_error'
scoring2 = 'neg_mean_absolute_error'
scoring3 = 'r2'
scoring4 = 'neg_mean_squared_error'
# endregion

# Prepare algorithms for Spot Checking
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('Ridge', Ridge()))

models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# Evaluation of various models and printing results
# Variations of Regressors * Variations of Error Metrics
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=k_folds, random_state=seed)
    res_1 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    res_2 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring2)
    res_3 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring3)
    res_4 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring4)
    msg = "%s: max error: %f , mean absolute error: %f, r2: %f, mean squared error: %f" % (name, res_1.mean(),
                                                                                           -res_2.mean(), res_3.mean(), -res_4.mean())
    print(msg)
