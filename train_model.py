# ============================================================
# Project 2: AI Order Status Classification
# File: train_model.py
# ============================================================

from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Current project folder ka path
BASE_DIR = Path(__file__).resolve().parent

# Folders ke paths
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"

# Agar folders nahi hain to automatically create ho jayenge
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. DATASET FILE FIND KARNA
# ============================================================

def find_dataset_file():
    """
    Pehle orders_dataset.xlsx file search karega.
    Agar exact name na mile to dataset folder ki pehli Excel file use karega.
    """

    expected_file = DATASET_DIR / "orders_dataset.xlsx"

    if expected_file.exists():
        return expected_file

    excel_files = list(DATASET_DIR.glob("*.xlsx"))

    if len(excel_files) == 1:
        print(
            f"Note: orders_dataset.xlsx nahi mili.\n"
            f"Is liye ye Excel file use ki ja rahi hai: {excel_files[0].name}"
        )
        return excel_files[0]

    if len(excel_files) == 0:
        raise FileNotFoundError(
            "Dataset folder mein koi .xlsx file nahi mili.\n"
            "Apni Excel file ko dataset folder ke andar rakho."
        )

    raise FileNotFoundError(
        "Dataset folder mein multiple Excel files hain.\n"
        "Required file ka naam orders_dataset.xlsx rakho."
    )


# ============================================================
# 3. FIXED DATASET COLUMNS
# ============================================================

TARGET_COLUMN = "OrderStatus"

FEATURE_COLUMNS = [
    "Product",
    "Quantity",
    "UnitPrice",
    "PaymentMethod",
    "ItemsInCart",
    "CouponCode",
    "ReferralSource",
    "TotalPrice"
]

CATEGORICAL_COLUMNS = [
    "Product",
    "PaymentMethod",
    "CouponCode",
    "ReferralSource"
]

NUMERICAL_COLUMNS = [
    "Quantity",
    "UnitPrice",
    "ItemsInCart",
    "TotalPrice"
]

REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


# ============================================================
# 4. DATASET LOAD KARNA
# ============================================================

dataset_path = find_dataset_file()

print("\n================================================")
print("DATASET LOADING")
print("================================================")
print(f"Dataset file: {dataset_path}")

df = pd.read_excel(dataset_path)

print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")


# ============================================================
# 5. EXACT COLUMNS VERIFY KARNA
# ============================================================

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Dataset mein ye required columns nahi mile:\n"
        f"{missing_columns}\n\n"
        "Excel file ke column names exact hone chahiye."
    )

print("\nRequired columns successfully verified.")


# ============================================================
# 6. REQUIRED DATA SELECT KARNA
# ============================================================

# Sirf model ke required columns select kar rahe hain
data = df[REQUIRED_COLUMNS].copy()


# ============================================================
# 7. DATA CLEANING
# ============================================================

# CouponCode mein 309 blank values hain.
# Website par blank coupon ko No Coupon kaha jayega.
data["CouponCode"] = data["CouponCode"].fillna("No Coupon")

# Text columns ke start/end spaces remove karna
for column in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
    data[column] = data[column].astype(str).str.strip()

# Numerical columns ko proper numeric form mein convert karna
for column in NUMERICAL_COLUMNS:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

# Agar conversion ke baad koi missing numerical value aaye
# to us column ke median se fill kar denge
for column in NUMERICAL_COLUMNS:
    data[column] = data[column].fillna(
        data[column].median()
    )

# Duplicate rows remove karna
before_duplicates = len(data)
data = data.drop_duplicates()
after_duplicates = len(data)

print(
    f"Duplicate rows removed: "
    f"{before_duplicates - after_duplicates}"
)


# ============================================================
# 8. DATASET INFORMATION SHOW KARNA
# ============================================================

print("\n================================================")
print("DATASET INFORMATION")
print("================================================")

print("\nTarget classes:")
print(data[TARGET_COLUMN].value_counts())

print("\nMissing values after cleaning:")
print(data.isnull().sum())


# ============================================================
# 9. INPUT FEATURES AUR TARGET
# ============================================================

X = data[FEATURE_COLUMNS]
y = data[TARGET_COLUMN]

print("\nFeature columns:")
for feature in FEATURE_COLUMNS:
    print(f"- {feature}")

print(f"\nTarget column: {TARGET_COLUMN}")


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n================================================")
print("TRAIN-TEST SPLIT")
print("================================================")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# ============================================================
# 11. DATA PREPROCESSING
# ============================================================

# Text columns ko One-Hot Encoding se numbers mein convert karega
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

# Numerical columns ko same scale par lane ke liye
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_COLUMNS
        ),
        (
            "numerical",
            numerical_transformer,
            NUMERICAL_COLUMNS
        )
    ]
)


# ============================================================
# 12. CLASSIFICATION ALGORITHMS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=8
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10
    )
}


# ============================================================
# 13. MODELS TRAIN AUR COMPARE KARNA
# ============================================================

results = []

best_model_name = None
best_pipeline = None
best_predictions = None
best_f1_score = -1

print("\n================================================")
print("MODEL TRAINING")
print("================================================")

for model_name, classifier in models.items():

    # Preprocessing aur classifier ko ek pipeline mein jorna
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    print(f"\nTraining: {model_name}")

    # Model training
    pipeline.fit(X_train, y_train)

    # Testing data ki predictions
    predictions = pipeline.predict(X_test)

    # Evaluation scores
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Best model weighted F1 score ke basis par select hoga
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = model_name
        best_pipeline = pipeline
        best_predictions = predictions


# ============================================================
# 14. MODEL COMPARISON SAVE KARNA
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

comparison_file = REPORTS_DIR / "model_comparison.csv"
results_df.to_csv(comparison_file, index=False)

print("\n================================================")
print("MODEL COMPARISON")
print("================================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ============================================================
# 15. BEST MODEL REPORT
# ============================================================

print("\n================================================")
print("BEST MODEL")
print("================================================")

print(f"Selected model: {best_model_name}")
print(f"Best weighted F1 score: {best_f1_score:.4f}")

report = classification_report(
    y_test,
    best_predictions,
    zero_division=0
)

print("\nClassification Report:")
print(report)

report_file = REPORTS_DIR / "classification_report.txt"

with open(report_file, "w", encoding="utf-8") as file:
    file.write(f"Best Model: {best_model_name}\n\n")
    file.write(report)


# ============================================================
# 16. CONFUSION MATRIX SAVE KARNA
# ============================================================

class_names = sorted(y.unique().tolist())

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    display_labels=class_names,
    xticks_rotation=45
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()

confusion_matrix_file = (
    REPORTS_DIR / "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. BEST TRAINED MODEL SAVE KARNA
# ============================================================

model_file = MODEL_DIR / "order_status_model.pkl"

joblib.dump(
    best_pipeline,
    model_file
)


# ============================================================
# 18. WEBSITE KE LIYE MODEL INFORMATION SAVE KARNA
# ============================================================

model_information = {
    "model_name": best_model_name,
    "target_column": TARGET_COLUMN,
    "feature_columns": FEATURE_COLUMNS,
    "categorical_columns": CATEGORICAL_COLUMNS,
    "numerical_columns": NUMERICAL_COLUMNS,
    "classes": class_names,

    "product_options": sorted(
        data["Product"].unique().tolist()
    ),

    "payment_method_options": sorted(
        data["PaymentMethod"].unique().tolist()
    ),

    "coupon_code_options": sorted(
        data["CouponCode"].unique().tolist()
    ),

    "referral_source_options": sorted(
        data["ReferralSource"].unique().tolist()
    )
}

metadata_file = MODEL_DIR / "model_information.json"

with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(
        model_information,
        file,
        indent=4
    )


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n================================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================================")

print(f"Saved model: {model_file}")
print(f"Model information: {metadata_file}")
print(f"Model comparison: {comparison_file}")
print(f"Classification report: {report_file}")
print(f"Confusion matrix: {confusion_matrix_file}")

