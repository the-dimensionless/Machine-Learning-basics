# classifying as view as good or bad

# X1, X2 are features (words in review, time when posted)
# 3d hyperplane: W1X1 + W2X2 + b <=> 0
# >0 -> positive, <=0 -> neagtive
# optimization problem to find W1, W2 and b

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

twenty_train = fetch_20newsgroups(subset="train", shuffle=True)

count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(twenty_train.data)

tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)

# specify normalization and tolerance
clf_svc = LinearSVC(penalty="12", dual=False, tol=1e-3)
clf_svc.fit(X_train_tfidf, twenty_train.target)

clf_svc_pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", LinearSVC(penalty="12", dual=False, tol=0.001)),
    ]
)

clf_svc_pipeline.fit(twenty_train.data, twenty_train.target)
twenty_test = fetch_20newsgroups(subset="test", shuffle=True)

predicted = clf_svc_pipeline.predict(twenty_test.data)

acc_svm = accuracy_score(twenty_test.target, predicted)
print("accuracy ", acc_svm)
