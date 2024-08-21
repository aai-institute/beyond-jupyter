# One monolithic function

Meet the One Monolithic Function, also known as the Swiss Army Knife Function or the God Function—the Jack-of-all-trades and master of, well, none.
These functions usually come with deceptively simple and generic names like *run*, *train*, or *main*. It may be the result of converting a Jupyter notebook, similar to what we’ve done in our [refactoring journey](../../refactoring-journey/step01-python-script/run_classifier_evaluation.py).

This function is like that one person who insists on doing everything themselves—from cooking dinner to fixing the plumbing—except in the coding world. But just like our multitasking friend who forgets to turn off the stove while unclogging the sink, this approach can quickly become a recipe for disaster.

By trying to do everything in one place, this monolithic function ends up being an unmaintainable tangle of responsibilities. It mixes high-level decisions like "What file format am I dealing with?" with low-level tasks like "Let's calculate the mean to fill in missing values," all in the same breath. It's a classic case of not knowing when to delegate, resulting in code that’s harder to read, harder to debug, and way harder to extend.

Take a look at the following code and try to understand what it does without reading it line by line:

```python
import pandas as pd
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main(file_path: str) -> float:
    
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
            data = pd.DataFrame(data_dict)
    else:
        raise ValueError("Unsupported file format!")

    logging.info(f"Data loaded from {file_path} with {len(data)} rows and {len(data.columns)} columns.")
    
    if 'target' not in data.columns:
        raise ValueError("Target column is missing in the dataset!")

    for column in data.columns:
        if data[column].isnull().sum() > 0:
            mean_value = data[column].mean()
            data[column].fillna(mean_value, inplace=True)
    
    for column in data.select_dtypes(include=[np.number]).columns:
        max_value = data[column].max()
        min_value = data[column].min()
        data[column] = (data[column] - min_value) / (max_value - min_value)
    
    data['feature_interaction'] = data['feature1'] * data['feature2'] * np.log1p(data['feature3'])

    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return accuracy
```
Sure, if you painstakingly read it line by line, you’ll eventually arrive at this thrilling revelation about what the function does:

1. Loads the data by handling different file formats.
2. Cleans the data by filling in missing values.
3. Normalizes the data through scaling.
4. Performs feature engineering by creating interaction terms.
5. Trains a machine learning model.
6. Evaluates model performance.

While this function does manage to accomplish several tasks, it suffers from poor readability. The mixture of different responsibilities within a single function makes it difficult to follow what’s happening at a glance.

Additionally, the function is not easily testable and hard to modify or extend. Because it handles everything from data loading to model evaluation, testing individual parts of the process in isolation is nearly impossible.
Any changes to one part of the process could potentially impact the others, making the function fragile and prone to errors when updates are needed.

A first entry point for refactoring this function could be to apply the [Single Level of Abstraction Principle (SLAP)](../../oop-essentials/03-general-principles/README.md/#slap-single-level-of-abstraction-principle). By ensuring that each function operates at a single level of abstraction, you can begin to separate the high-level orchestration from the low-level details. The result could look like this:

```python
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main(path: str) -> float:
    data = load_data(path)
    data = fill_missing_values(data)
    data = normalize_features(data)
    data = engineer_features(data)
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy


def load_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
            data = pd.DataFrame(data_dict)
    else:
        raise ValueError("Unsupported file format!")
    logging.info(f"Data loaded from {file_path} with {len(data)} rows and {len(data.columns)} columns.")
    return data


def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            mean_value = data[column].mean()
            data[column].fillna(mean_value, inplace=True)
    return data


def normalize_features(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.select_dtypes(include=[np.number]).columns:
        max_value = data[column].max()
        min_value = data[column].min()
        data[column] = (data[column] - min_value) / (max_value - min_value)
    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data['feature_interaction'] = data['feature1'] * data[
        'feature2'] * np.log1p(data['feature3'])
    return data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    y = data['target']
    X = data.drop('target', axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return accuracy

```
By simply extracting low-level functions in a way that the low-level functions have an isolated task and calling
them in the high-level function *main*, we already gained:

1. Improved Readability: The main function now reads more like a summary of the overall process, with each low-level function clearly named to describe its specific task. This makes it easier for developers to understand the code at a glance.

2. Enhanced Testability: Isolated functions are easier to unit test. 
 You can test each low-level function individually to ensure it performs its task correctly, leading to more reliable and robust code.

3. Increased Reusability: Low-level functions that perform specific tasks can often be reused in different parts of the codebase or in future projects, reducing the need to write redundant code.

Nevertheless, this should only be considered a first step toward improving the code. While extracting low-level functions helps provide more clarity about what’s happening, the code still lacks a coherent software design and remains fragile and inflexible. To achieve a truly robust and maintainable solution, further refactoring is necessary.

We highly encourage you to follow our [refactoring journey](../../refactoring-journey/README.md) to explore a more structured and well-designed approach. This will not only enhance the code’s flexibility but also ensure it’s better suited to handle future changes and extensions.