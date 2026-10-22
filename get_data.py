"""Downloads the Telco Customer Churn dataset from Kaggle into this folder."""

import shutil
from pathlib import Path

import kagglehub

path = kagglehub.dataset_download("blastchar/telco-customer-churn")
src = Path(path) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
dst = Path(__file__).resolve().parent / src.name
shutil.copy(src, dst)
print("Done:", dst)
