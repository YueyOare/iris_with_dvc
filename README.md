# Iris Classification with DVC

A machine learning project for classifying Iris flower species using scikit-learn and DVC (Data Version Control) for experiment tracking and pipeline management.

## 📋 Project Overview

This project implements and compares two machine learning models for classifying Iris flower species:
- **Logistic Regression** with elastic net regularization
- **Random Forest** classifier

The project uses DVC to manage the ML pipeline, track experiments, and version control data and models.

## 🎯 Features

- **Automated ML Pipeline**: Data preparation, model training, and evaluation stages
- **Model Comparison**: Side-by-side comparison of Logistic Regression and Random Forest
- **Experiment Tracking**: Using DVCLive for metrics and plots
- **Hyperparameter Management**: Centralized configuration via `params.yaml`
- **Reproducible Results**: DVC ensures reproducible experiments
- **Visual Analytics**: Confusion matrices and feature importance plots

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- DVC
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd iris_with_dvc
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # You may need to create this file
   # Or install manually:
   pip install pandas scikit-learn pyyaml python-box joblib matplotlib dvclive dvc
   ```

4. **Initialize DVC** (if not already done):
   ```bash
   dvc init
   ```

5. **Pull the data**:
   ```bash
   dvc pull
   ```

## 📊 Usage

### Run the Complete Pipeline

Execute the entire ML pipeline with a single command:

```bash
dvc repro
```

This will run all stages in sequence:
1. Data preparation and splitting
2. Logistic Regression training
3. Random Forest training
4. Model evaluation and comparison

### Run Individual Stages

You can also run individual pipeline stages:

```bash
# Data preparation only
dvc repro prepare

# Train specific model
dvc repro logistic_regression
dvc repro random_forest

# Evaluation only
dvc repro evaluate
```

### View Results

**Check metrics**:
```bash
dvc metrics show
```

**View plots**:
```bash
dvc plots show
```

**Compare experiments**:
```bash
dvc exp show
```

## 📁 Project Structure

```
iris_with_dvc/
├── data/
│   ├── iris.csv.dvc          # DVC-tracked iris dataset
│   └── prepared/             # Processed data splits
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/                   # Trained models
│   ├── logreg.joblib
│   └── randforest.joblib
├── eval/                     # Evaluation results
│   ├── metrics.json
│   └── plots/
├── dvc_plots/               # DVC plot configurations
├── prepare.py               # Data preparation script
├── logistic_regression.py   # Logistic regression training
├── random_forest.py         # Random forest training
├── evaluate.py              # Model evaluation script
├── params.yaml              # Hyperparameters configuration
├── dvc.yaml                 # DVC pipeline definition
├── dvc.lock                 # DVC lock file
└── README.md
```

## ⚙️ Configuration

Modify hyperparameters in `params.yaml`:

```yaml
prepare:
    data_path: data/iris.csv
    test_size: 0.3
    
logistic_regression:
    penalty: elasticnet
    c: 0.5
    l1_ratio: 0.5
    solver: saga
    multi_class: multinomial
    
random_forest:
    n_estimators: 10
    max_depth: 3
    min_samples_split: 2
    criterion: gini

evaluate:
    eval_path: eval
```

## 🔄 Pipeline Stages

### 1. Data Preparation (`prepare.py`)
- Loads the Iris dataset
- Performs train/test split (70/30 by default)
- Applies standard scaling to features
- Saves preprocessed data splits

### 2. Model Training
- **Logistic Regression** (`logistic_regression.py`): Trains with elastic net regularization
- **Random Forest** (`random_forest.py`): Trains ensemble classifier

### 3. Evaluation (`evaluate.py`)
- Calculates performance metrics (ROC-AUC, Average Precision)
- Generates confusion matrices
- Creates feature importance plots (for Random Forest)
- Compares models on both training and test sets

## 📈 Results and Metrics

The evaluation generates:
- **ROC-AUC scores** for multi-class classification
- **Average Precision scores** 
- **Confusion matrices** for both models
- **Feature importance plots** for Random Forest
- **Train vs Test performance** comparison

Access results via:
```bash
# View numeric metrics
cat eval/metrics.json

# View DVC metrics dashboard
dvc studio  # Opens web interface
```

## 🔧 Development

### Experiment with Different Parameters

1. Modify `params.yaml` with new hyperparameters
2. Run the pipeline: `dvc repro`
3. Compare results: `dvc exp show`

### Add New Models

1. Create a new training script (e.g., `svm.py`)
2. Add the stage to `dvc.yaml`
3. Update `evaluate.py` to include the new model

### Track Experiments

```bash
# Create experiment branch
dvc exp run -n "experiment_name"

# Compare experiments
dvc exp diff
dvc exp show --no-pager
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the pipeline to ensure everything works
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) from UCI ML Repository
- [DVC](https://dvc.org/) for data version control and ML pipeline management
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
