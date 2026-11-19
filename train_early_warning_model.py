import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 75)
    print("   SAARTHI EARLY-WARNING MODEL (1ST SEMESTER ONLY) - EVALUATION   ")
    print("=" * 75)

    # 1. Load the CSV
    data_path = os.path.join("data", "uci_dropout.csv")
    print(f"\n[1] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)

    # 2. Drop rows where Target is 'Enrolled'
    print("\n[2] Filtering dataset: Dropping 'Enrolled' records...")
    df_filtered = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    df_filtered["Target"] = df_filtered["Target"].map({"Dropout": 1, "Graduate": 0})

    # 3. Select Only 1st Semester + Demographic & Financial Features (Exclude 2nd Sem)
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

    print("\n[3] Selected Early-Warning Features (10 features, 2nd Sem excluded):")
    for idx, col in enumerate(early_warning_features, 1):
        print(f"    {idx:2d}. {col}")

    X = df_filtered[early_warning_features]
    y = df_filtered["Target"]

    print(f"\n    Total samples: {len(y)} (Dropout: {y.sum()}, Graduate: {len(y) - y.sum()})")

    # 4. Train / Test Split (80/20 stratified split, random_state=42)
    print("\n[4] Splitting dataset into train (80%) and test (20%) [stratified, random_state=42]...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Training set size: {X_train.shape[0]}")
    print(f"    Testing set size:  {X_test.shape[0]}")

    # 5. Train Models
    print("\n[5] Training Early-Warning Models...")

    # (a) Rule-Based Baseline
    def rule_predict(X_subset):
        diff = X_subset["Curricular units 1st sem (enrolled)"] - X_subset["Curricular units 1st sem (approved)"]
        return (diff > 2).astype(int)

    print("    (a) Rule-based Baseline configured.")

    # (b) Logistic Regression
    print("    (b) Training Logistic Regression...")
    lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    lr_model.fit(X_train, y_train)

    # (c) Gradient Boosting Classifier
    print("    (c) Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)

    # 6. Evaluation on Test Set
    print("\n[6] Evaluating Early-Warning Models on test set...")

    y_pred_rule = rule_predict(X_test)
    y_proba_rule = y_pred_rule.values.astype(float)

    y_pred_lr = lr_model.predict(X_test)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

    y_pred_gb = gb_model.predict(X_test)
    y_proba_gb = gb_model.predict_proba(X_test)[:, 1]

    def compute_metrics(y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        pr_auc = average_precision_score(y_true, y_proba)
        return {
            "Accuracy": acc,
            "Recall": rec,
            "Precision": prec,
            "F2 Score": f2,
            "PR-AUC": pr_auc,
        }

    metrics_rule = compute_metrics(y_test, y_pred_rule, y_proba_rule)
    metrics_lr = compute_metrics(y_test, y_pred_lr, y_proba_lr)
    metrics_gb = compute_metrics(y_test, y_pred_gb, y_proba_gb)

    # 7. Precision@30 for Early-Warning Gradient Boosting
    test_df_gb = pd.DataFrame({"y_true": y_test.values, "proba": y_proba_gb})
    top_30_gb = test_df_gb.sort_values(by="proba", ascending=False).head(30)
    precision_at_30 = top_30_gb["y_true"].mean()
    top_30_dropouts = int(top_30_gb["y_true"].sum())

    print(f"\n[7] Early-Warning Gradient Boosting Precision@30:")
    print(f"    Top 30 predicted highest-risk students actual dropouts: {top_30_dropouts}/30")
    print(f"    Precision@30 = {precision_at_30:.4f} ({precision_at_30 * 100:.1f}%)")

    # 8. Comparison Table
    print("\n[8] Early-Warning Model Performance Comparison Table (Test Set):")
    print("-" * 75)
    header = f"{'Metric':<18} | {'Rule-Based Baseline':<20} | {'Logistic Regression':<20} | {'Gradient Boosting':<18}"
    print(header)
    print("-" * 75)

    metric_names = ["Accuracy", "Recall", "Precision", "F2 Score", "PR-AUC"]
    for m in metric_names:
        r_val = f"{metrics_rule[m]:.4f} ({metrics_rule[m]*100:.1f}%)"
        lr_val = f"{metrics_lr[m]:.4f} ({metrics_lr[m]*100:.1f}%)"
        gb_val = f"{metrics_gb[m]:.4f} ({metrics_gb[m]*100:.1f}%)"
        print(f"{m:<18} | {r_val:<20} | {lr_val:<20} | {gb_val:<18}")

    print("-" * 75)

    # Save early warning model as model_early_warning.pkl
    joblib.dump(gb_model, "model_early_warning.pkl")
    print("\n[Saved] Early-warning Gradient Boosting model saved to 'model_early_warning.pkl'")

    # 9. SHAP Feature Importance (Top 8 features)
    print("\n[9] Computing SHAP Feature Importance for Early-Warning Gradient Boosting model...")
    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    shap_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Mean |SHAP Value|": mean_abs_shap}
    ).sort_values(by="Mean |SHAP Value|", ascending=False)

    print("\n" + "=" * 75)
    print("   TOP 8 FEATURES BY MEAN ABSOLUTE SHAP VALUE (EARLY-WARNING MODEL)   ")
    print("=" * 75)
    top_8_shap = shap_importance_df.head(8)
    for rank, (_, row) in enumerate(top_8_shap.iterrows(), 1):
        print(f"   {rank}. {row['Feature']:<45} : {row['Mean |SHAP Value|']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
