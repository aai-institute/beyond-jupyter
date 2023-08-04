from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from songpop.data import Dataset
from songpop.model_factory import ModelFactory


def main():
    # define & load dataset
    dataset = Dataset(10000)
    X, y = dataset.load_xy()

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, shuffle=True)

    # define models to be evaluated
    models = [
        ModelFactory.create_logistic_regression_orig(),
        ModelFactory.create_knn_orig(),
        ModelFactory.create_random_forest_orig(),
        ModelFactory.create_decision_tree_orig(),
    ]

    # evaluate models
    for model in models:
        print(f"Evaluating model:\n{model}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
