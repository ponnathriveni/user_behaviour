# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import joblib

# ==============================
# LOAD DATA
# ==============================
# Make sure filename is correct!
data = pd.read_csv("ecommerce_data.csv")
# ==============================
# BASIC INFO
# ==============================
print("First 5 rows:\n", data.head())
print("\nMissing values:\n", data.isnull().sum())

# ==============================
# HANDLE MISSING VALUES
# ==============================
data = data.dropna()

# ==============================
# ENCODE CATEGORICAL COLUMNS
# ==============================
label_cols = data.select_dtypes(include='object').columns

for col in label_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# ==============================
# DEFINE FEATURES & TARGET
# ==============================
# Change 'purchase' if your target column name is different
if 'purchase' not in data.columns:
    raise Exception("❌ 'purchase' column not found. Check your dataset.")

X = data.drop('purchase', axis=1)
y = data['purchase']

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TRAIN MODEL
# ==============================
model = LogisticRegression()
model.fit(X_train, y_train)
# PREDICTIONS
# ==============================
y_pred = model.predict(X_test)

# ==============================
# EVALUATION
# ==============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, "ecommerce_model.pkl")

print("\n✅ Model saved as ecommerce_model.pkl")