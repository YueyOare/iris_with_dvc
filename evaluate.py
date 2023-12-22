import os
import yaml

import pandas as pd
from sklearn import metrics
from dvclive import Live
from matplotlib import pyplot as plt
from box import ConfigBox
from yaml.loader import SafeLoader
import joblib
from sklearn.preprocessing import label_binarize


def evaluate(model, X, y, split, model_name, live, save_path):
    predictions_by_class = model.predict_proba(X)
    y_binarized = label_binarize(y, classes=model.classes_)
    avg_prec_scores = []
    for i in range(y_binarized.shape[1]):
        avg_prec = metrics.average_precision_score(y_binarized[:, i], predictions_by_class[:, i])
        avg_prec_scores.append(avg_prec)

    # Average over all classes
    avg_prec = sum(avg_prec_scores) / len(avg_prec_scores)

    roc_auc = metrics.roc_auc_score(y_binarized, predictions_by_class, average='macro')
    if model_name not in live.summary:
        live.summary[model_name] = {"avg_prec": {}, "roc_auc": {}}

    live.summary[model_name]["avg_prec"][split] = avg_prec
    live.summary[model_name]["roc_auc"][split] = roc_auc

    live.log_sklearn_plot(
        "confusion_matrix",
        y.squeeze(),
        predictions_by_class.argmax(-1),
        name=f"cm/{model_name}_{split}",
    )


def save_importance_plot(live, model, model_name, feature_names):
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel(f"{model_name}: Feature Importance")

    # Check if the model has the attribute 'feature_importances_'
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        forest_importances = pd.Series(importances, index=feature_names).nlargest(n=30)
        forest_importances.plot.bar(ax=axes)
        live.log_image(f"importance_{model_name}.png", fig)
    else:
        return


def main():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8"), Loader=SafeLoader))
    eval_path = params.evaluate.eval_path
    # Load data
    X_train = pd.read_csv('data/prepared/X_train.csv')
    y_train = pd.read_csv('data/prepared/y_train.csv').iloc[:, 0].to_numpy()
    X_test = pd.read_csv('data/prepared/X_test.csv')
    y_test = pd.read_csv('data/prepared/y_test.csv').iloc[:, 0].to_numpy()

    # Load models
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    loaded_models = []
    with Live(eval_path, dvcyaml=False) as live:
        for model_file in model_files:
            model_path = os.path.join('models', model_file)
            model = joblib.load(model_path)
            loaded_models.append(model)
            model_name = model_file.split('.')[0]

            # Evaluate train and test datasets.
            evaluate(model, X_train, y_train, "train", model_name, live, save_path=eval_path)
            evaluate(model, X_test, y_test, "test", model_name, live, save_path=eval_path)

            # Dump feature importance plot.
            save_importance_plot(live, model, model_name, X_train.columns)

    print('done')


if __name__ == "__main__":
    main()
