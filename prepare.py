import pandas as pd
import yaml
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from box import ConfigBox
from yaml.loader import SafeLoader

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8"), Loader=SafeLoader))

data_path = params.prepare.data_path
test_size = params.prepare.test_size

data = pd.read_csv(data_path)

X = data.drop('variety', axis=1)
y = data['variety']
encoder = LabelEncoder().fit_transform(y)
features = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

scaler = StandardScaler().fit(X_train)
X_train_standardized = pd.DataFrame(scaler.transform(X_train), columns=features)
X_test_standardized = pd.DataFrame(scaler.transform(X_test), columns=features)

X_train_standardized.to_csv('data/X_train.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
X_test_standardized.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
