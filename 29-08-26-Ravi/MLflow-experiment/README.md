## This mlflow directory is creater the accuracy and performance of xgboost and pytorch_lstm model for the classification.

### Experiment-directory structure.
```text
TBM_Fraud_Detection/
│
├── data/
│
├── models/
│
├── artifacts/
│
├── reports/
│   └── figures/
│
├── src/
│
├── notebooks/
│
├── mlruns/
│
├── train.py
├── quantize_lstm.py
├── requirements.txt
└── README.md
```

### Install the requirements.txt.
```text
pip install requirements.txt
```

### Run the train.py

### run 
```text
mlflow ui --backend-store-uri sqlite:///mlflow.db
```