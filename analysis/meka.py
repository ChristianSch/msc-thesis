from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from skmultilearn.ext import Meka

X, y = make_multilabel_classification(sparse = True,
    return_indicator = 'sparse')

X_train, X_test, y_train, y_test = train_test_split(X,
        y,
        test_size=0.33)

"""
meka_classifier = "meka.classifiers.multilabel.PCC",
weka_classifier = "weka.classifiers.functions.Logistic",
"""
meka = Meka(
    meka_classifier = "meka.classifiers.multilabel.LC",
    weka_classifier = "weka.classifiers.bayes.NaiveBayes",
    meka_classpath = "/home/loki/Downloads/meka-release-1.9.3-SNAPSHOT/lib/",
    java_command = "/usr/bin/java")

try:
    meka.fit(X_train, y_train)
except Exception as e:
    print(e)
    print(e.args[0].decode('utf8'))



predictions = meka.predict_proba(X_test)

# hamming_loss(y_test, predictions)

print(predictions)
