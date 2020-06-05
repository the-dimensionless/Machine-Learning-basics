import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
import numpy as np

titanic_data = pd.read_csv("../data/titanic.csv", quotechar='"')
titanic_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], "columns", inplace=True)

le = preprocessing.LabelEncoder()
titanic_data["Sex"] = le.fit_transform(titanic_data["Sex"].astype(str))

titanic_data = pd.get_dummies(titanic_data, columns=["Embarked"])

titanic_data = titanic_data.dropna()

# bandwith -> kernel dependency
analyzer = MeanShift(bandwidth=50)
analyzer.fit(titanic_data)

# get an estimate of good bandwith ~ expecting 30
estimate_bandwidth(titanic_data)

labels = analyzer.labels_

# let's display number of clusters
print(np.unique(labels))

titanic_data["cluster_group"] = np.nan
data_length = len(titanic_data)
for i in range(data_length):
    titanic_data.iloc[i, titanic_data.columns.get_loc("cluster_group")] = labels[i]

# titanic_data.describe()

titanic_cluster_data = titanic_data.groupby(["cluster_group"]).mean()
titanic_cluster_data["Counts"] = pd.Series(
    titanic_data.groupby(["cluster_group"]).size()
)

print(titanic_cluster_data)
