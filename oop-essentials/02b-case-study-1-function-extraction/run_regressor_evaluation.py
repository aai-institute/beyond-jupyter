from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from songpop.data import *


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print(f"{model}: MAE={mae:.1f}")


def main():
    dataset = Dataset(10000)
    X, y = dataset.load_xy_projected_scaled()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, shuffle=True)

    # evaluate models
    evaluate_model(LogisticRegression(solver='lbfgs', max_iter=1000), X_train, y_train, X_test, y_test)
    evaluate_model(KNeighborsRegressor(n_neighbors=1), X_train, y_train, X_test, y_test)
    evaluate_model(RandomForestRegressor(n_estimators=100), X_train, y_train, X_test, y_test)
    evaluate_model(DecisionTreeRegressor(random_state=42, max_depth=2), X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
