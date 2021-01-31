import mlflow
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_experiment("/scklearn-handwrite")

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

mlflow.sklearn.autolog()
gamma_list = [10 ** x for x in range(-10, 10)]
for gamma in gamma_list:
    with mlflow.start_run():
        clf = svm.SVC(gamma=gamma)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)

        mlflow.log_metric("accuracy_score", accuracy_score(predicted, y_test))

