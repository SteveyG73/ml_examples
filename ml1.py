# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def get_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)
    return dataset


def split_dataset(dataset, seed):
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.20
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)
    split_set = {'X_train': x_train, "X_validation": x_validation, "Y_train": y_train, "Y_validation": y_validation}
    return split_set


def main():
    seed = 7
    scoring = 'accuracy'

    dataset = get_dataset()
    # Split-out validation dataset
    split_set = split_dataset(dataset, seed)
    # Spot Check Algorithms
    models = [('LR', LogisticRegression()),
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('NB', GaussianNB()),
              ('SVM', SVC())]
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, split_set['X_train'], split_set['Y_train'], cv=kfold,
                                                     scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    svm = SVC()

    svm.fit(split_set['X_train'], split_set['Y_train'])

    predictions = svm.predict(split_set['X_validation'])

    print(accuracy_score(split_set['Y_validation'], predictions))
    print(confusion_matrix(split_set['Y_validation'], predictions))
    print(classification_report(split_set['Y_validation'], predictions))


if __name__ == '__main__':
    main()
