from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from songpop.data import *


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print(f"{model}: MAE={mae:.1f}")


def main():
    dataset = Dataset(10000)
    X, y = dataset.load_xy()

    # project and scale data
    cols_used_by_models = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS, COL_DURATION_MS]
    X = X[cols_used_by_models]
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, test_size=0.3, shuffle=True)

    # evaluate models
    evaluate_model(LogisticRegression(solver='lbfgs', max_iter=1000), X_train, y_train, X_test, y_test)
    evaluate_model(KNeighborsRegressor(n_neighbors=1), X_train, y_train, X_test, y_test)
    evaluate_model(RandomForestRegressor(n_estimators=100), X_train, y_train, X_test, y_test)
    evaluate_model(DecisionTreeRegressor(random_state=42, max_depth=2), X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
