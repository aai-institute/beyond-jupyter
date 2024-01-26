import config
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    spotify_tracks = pd.read_csv(config.csv_data_path())

    # adding a new column in data set duration_mins by diving duration_ms (duration milliseconds) by 60000
    spotify_tracks['duration_mins'] = spotify_tracks['duration_ms'] / 60000

    # dropping the columns we don't need
    spotify_tracks = spotify_tracks.drop(['track_id', 'duration_ms'], axis=1)

    popularity_verdict = spotify_tracks.copy()

    # As almost 15% of entries have 0 popularity score, we drop the records with 0 popularity score as this will help
    # model in predicting better. 0 value records will not have significance in our analysis.
    popularity_verdict = popularity_verdict[popularity_verdict.popularity > 0]

    # setting ratings based on popularity score - popularity score 0 - 50 = Low, score = 51 - 100 = Popular
    popularity_verdict['verdict'] = ''
    for i, row in popularity_verdict.iterrows():
        score = 'low'
        if row.popularity >= 50:
            score = 'popular'
        popularity_verdict.at[i, 'verdict'] = score

    pop_ver_att = popularity_verdict[['year', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_mins']]

    # defining x and y df for our analysis
    X = pop_ver_att.select_dtypes(include='number')
    print(X.columns)
    y = popularity_verdict['verdict']

    scaler = StandardScaler()
    model_X = scaler.fit(X)
    X_scaled = model_X.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, test_size=0.3, shuffle=True)

    log_reg = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print("Logistic Regression Model Accuracy (in %):",
        metrics.accuracy_score(y_test, y_pred) * 100)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    print("KNN Model Accuracy (in %):",
        metrics.accuracy_score(y_test, pred) * 100)

    rforest = RandomForestClassifier(n_estimators=100)
    rforest.fit(X_train, y_train)
    y_pred = rforest.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Random Forest Model Accuracy (in %):",
        metrics.accuracy_score(y_test, y_pred) * 100)

    d_tree = DecisionTreeClassifier(random_state=42, max_depth=2)
    d_tree.fit(X_train, y_train)
    y_pred = d_tree.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Decsision Tree Model Accuracy (in %):",
        metrics.accuracy_score(y_test, y_pred) * 100)
