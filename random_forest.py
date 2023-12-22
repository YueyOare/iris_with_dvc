import pandas as pd
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from box import ConfigBox
from yaml.loader import SafeLoader

# Parameters defining
params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8"), Loader=SafeLoader))
n_estimators = params.random_forest.n_estimators
max_depth = params.random_forest.max_depth
min_samples_split = params.random_forest.min_samples_split
criterion = params.random_forest.criterion


X_train = pd.read_csv('data/prepared/X_train.csv')
y_train = pd.read_csv('data/prepared/y_train.csv').iloc[:, 0]
X_test = pd.read_csv('data/prepared/X_test.csv')
y_test = pd.read_csv('data/prepared/y_test.csv').iloc[:, 0]

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               criterion=criterion, n_jobs=-1)
model.fit(X_train, y_train)
print(f'Accuracy: {model.score(X_test, y_test)}')

joblib.dump(model, 'models/randforest_model.joblib')