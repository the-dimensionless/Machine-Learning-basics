# using mnist dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

mnist_data = pd.read_csv("../data/mnist/train.csv")
# print(mnist_data.tail())

features = mnist_data.columns[1:]
X = mnist_data[features]
Y = mnist_data["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X / 255.0, Y, test_size=0.1, random_state=0
)

svm = LinearSVC(penalty="12", dual=False, tol=1e-5)
svm.fit(X_train, Y_train)

y_predicted = svm.predict(X_test)
acc_svm = accuracy_score(Y_test, y_predicted)
print("SVM accuracy ", acc_svm)
