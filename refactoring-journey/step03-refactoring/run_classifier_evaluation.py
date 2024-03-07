from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from songpop.data import (Dataset, COL_YEAR, COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE,
                          COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS, COL_DURATION_MS)


def main():
    # define & load dataset
    dataset = Dataset(10000)
    X, y = dataset.load_xy()

    # project to columns used by models
    cols_used_by_models = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS, COL_DURATION_MS]
    X = X[cols_used_by_models]

    scaler = StandardScaler()
    model_X = scaler.fit(X)
    X_scaled = model_X.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, test_size=0.3, shuffle=True)

    # define models to be evaluated
    models = [
        linear_model.LogisticRegression(solver='lbfgs', max_iter=1000),
        KNeighborsClassifier(n_neighbors=1),
        RandomForestClassifier(n_estimators=100),
        DecisionTreeClassifier(random_state=42, max_depth=2)
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
