import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def main():
    print("=" * 70)
    print("   SAARTHI EARLY-WARNING MODEL: GENDER SUBGROUP FAIRNESS ANALYSIS   ")
    print("=" * 70)

    # 1. Load data
    data_path = os.path.join("data", "uci_dropout.csv")
    df = pd.read_csv(data_path)
    df_filtered = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    df_filtered["Target"] = df_filtered["Target"].map({"Dropout": 1, "Graduate": 0})

    early_warning_features = [
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Scholarship holder",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Age at enrollment",
    ]

    X = df_filtered[early_warning_features]
    y = df_filtered["Target"]

    # 2. Re-create exact 80/20 stratified split (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Load trained Early-Warning Gradient Boosting model
    gb_model = joblib.load("model_early_warning.pkl")
    y_pred = gb_model.predict(X_test)

    eval_df = X_test.copy()
    eval_df["y_true"] = y_test
    eval_df["y_pred"] = y_pred

    # 4. Calculate subgroup metrics
    results = []
    for g_val in [0, 1]:
        sub = eval_df[eval_df["Gender"] == g_val]
        tn, fp, fn, tp = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        accuracy = (tp + tn) / len(sub)

        results.append({
            "Gender": g_val,
            "Total (n)": len(sub),
            "Actual Dropouts": tp + fn,
            "Actual Graduates": tn + fp,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Recall": recall,
            "FPR": fpr,
            "Precision": precision,
            "Accuracy": accuracy,
        })

    # 5. Print Output Table
    print("\n" + "-" * 70)
    print(f"{'Gender Subgroup':<18} | {'Sample Size (n)':<16} | {'Recall (TPR)':<15} | {'False Positive Rate (FPR)':<25}")
    print("-" * 70)
    for r in results:
        g_name = f"Gender = {r['Gender']}"
        rec_str = f"{r['Recall']:.4f} ({r['Recall']*100:.2f}%)"
        fpr_str = f"{r['FPR']:.4f} ({r['FPR']*100:.2f}%)"
        n_str = f"{r['Total (n)']} (P={r['Actual Dropouts']}, N={r['Actual Graduates']})"
        print(f"{g_name:<18} | {n_str:<16} | {rec_str:<15} | {fpr_str:<25}")
    print("-" * 70)

    # Detailed Confusion Breakdown
    print("\nConfusion Matrix Breakdown:")
    for r in results:
        print(f" • Gender = {r['Gender']}: TP={r['TP']}, FP={r['FP']}, FN={r['FN']}, TN={r['TN']} | Precision={r['Precision']:.4f} | Accuracy={r['Accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
