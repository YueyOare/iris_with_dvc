import pandas as pd
import yaml
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from box import ConfigBox
from yaml.loader import SafeLoader

# Parameters defining
params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8"), Loader=SafeLoader))
penalty = params.logistic_regression.penalty
C = params.logistic_regression.c
multi_class = params.logistic_regression.multi_class
solver = params.logistic_regression.solver
l1_ratio = params.logistic_regression.l1_ratio

X_train = pd.read_csv('data/prepared/X_train.csv')
y_train = pd.read_csv('data/prepared/y_train.csv').iloc[:, 0]
X_test = pd.read_csv('data/prepared/X_test.csv')
y_test = pd.read_csv('data/prepared/y_test.csv').iloc[:, 0]

model = LogisticRegression(penalty=penalty, C=C, solver=solver,
                           multi_class=multi_class,
                           l1_ratio=l1_ratio, n_jobs=-1,
                           random_state=42)
model.fit(X_train, y_train)
print(f'Accuracy: {model.score(X_test, y_test)}')

joblib.dump(model, 'models/logreg.joblib')