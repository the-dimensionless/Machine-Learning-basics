import pandas as pd
import numpy as np

wine_data = pd.read_csv(
    "../data/winequality-white.csv",
    names=[
        "Fixed Acidity",
        "Volatile Acidity",
        "Citric Acid",
        "Residual Sugar",
        "Chlorides",
        "Free Sulfur Dioxide",
        "Total Sulfur Dioxide",
        "Density",
        "pH",
        "Sulphates",
        "Alcohol",
        "Quality",
    ],
    skiprows=1,
    sep=r"\s*;\s*",
    engine="python",
)
wine_data.head()
wine_data["Quality"].unique()

X = wine_data.drop("Quality", axis=1)
Y = wine_data["Quality"]

from sklearn import preprocessing

X = preprocessing.scale(X)
from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.svm import LinearSVC

clf_svc = LinearSVC(penalty="l1", dual=False, tol=1e-3)
clf_svc.fit(X_train, Y_train)

accuracy = clf_svc.score(x_test, y_test)
print(accuracy)

import matplotlib.pyplot as plt
import seaborn as sns

corrmat = wine_data.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.set(font_scale=1.1)
sns.heatmap(corrmat, vmax=0.8, square=True, annot=True, fmt=".2f", cmap="winter")
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=1, whiten=True)
X_reduced = pca.fit_transform(X)

# pca.explained_variance_
# pca.explained_variance_ratio_

import matplotlib.pyplot as plt

plt.plot(pca.explained_variance_ratio_)
plt.xlabel("Dimension")
plt.ylabel("Explain Variance Ratio")
plt.show()

X_train, x_test, Y_train, y_test = train_test_split(
    X_reduced, Y, test_size=0.2, random_state=0
)
clf_svc_pca = LinearSVC(penalty="l1", dual=False, tol=1e-3)
clf_svc_pca.fit(X_train, Y_train)

accuracy = clf_svc_pca.score(x_test, y_test)
print(accuracy)
