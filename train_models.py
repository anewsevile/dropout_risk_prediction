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
    print("=" * 70)
    print("   SAARTHI / DROPOUT RISK PREDICTION - MODEL TRAINING & EVALUATION   ")
    print("=" * 70)

    # 1. Load the CSV
    data_path = os.path.join("data", "uci_dropout.csv")
    print(f"\n[1] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"    Initial dataset shape: {df.shape}")

    # 2. Drop rows where Target is 'Enrolled' (keep only Dropout and Graduate)
    print("\n[2] Filtering dataset: Dropping 'Enrolled' records...")
    df_filtered = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    print(f"    Filtered dataset shape: {df_filtered.shape}")
    print(f"    Target class distribution:\n{df_filtered['Target'].value_counts().to_string()}")

    # 3. Encode Target as binary: Dropout = 1, Graduate = 0
    print("\n[3] Encoding Target (Dropout = 1, Graduate = 0)...")
    df_filtered["Target"] = df_filtered["Target"].map({"Dropout": 1, "Graduate": 0})

    X = df_filtered.drop(columns=["Target"])
    y = df_filtered["Target"]

    print(f"    Feature count: {X.shape[1]}")
    print(f"    Total samples: {len(y)} (Dropout: {y.sum()}, Graduate: {len(y) - y.sum()})")

    # 4. Train / Test Split (80/20 stratified split, random_state=42)
    print("\n[4] Splitting dataset into train (80%) and test (20%) [stratified, random_state=42]...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Training set size: {X_train.shape[0]}")
    print(f"    Testing set size:  {X_test.shape[0]}")

    # 5. Train Models
    print("\n[5] Training models...")

    # (a) Rule-Based Baseline
    # Flags Dropout (1) if 'Curricular units 1st sem (approved)' < 'Curricular units 1st sem (enrolled)' by more than 2
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
    print("\n[6] Evaluating models on test set...")

    # Rule predictions & scores
    y_pred_rule = rule_predict(X_test)
    y_proba_rule = y_pred_rule.values.astype(float)

    # Logistic Regression predictions & probabilities
    y_pred_lr = lr_model.predict(X_test)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

    # Gradient Boosting predictions & probabilities
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

    # 7. Precision@30 for Gradient Boosting
    test_df_gb = pd.DataFrame({"y_true": y_test.values, "proba": y_proba_gb})
    top_30_gb = test_df_gb.sort_values(by="proba", ascending=False).head(30)
    precision_at_30 = top_30_gb["y_true"].mean()
    top_30_dropouts = int(top_30_gb["y_true"].sum())

    print(f"\n[7] Gradient Boosting Precision@30:")
    print(f"    Top 30 predicted highest-risk students actual dropouts: {top_30_dropouts}/30")
    print(f"    Precision@30 = {precision_at_30:.4f} ({precision_at_30 * 100:.1f}%)")

    # 8. Comparison Table
    print("\n[8] Model Performance Comparison Table (Test Set):")
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

    # 9. Save Gradient Boosting Model
    model_output_path = "model.pkl"
    print(f"\n[9] Saving trained Gradient Boosting model to '{model_output_path}'...")
    joblib.dump(gb_model, model_output_path)
    print(f"    Model successfully saved ({os.path.getsize(model_output_path) / 1024:.1f} KB).")

    # 10. SHAP Feature Importance (Top 8 features)
    print("\n[10] Computing SHAP Feature Importance for Gradient Boosting model...")
    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(X_test)

    # shap_values can be 2D array (n_samples, n_features) or list for binary
    if isinstance(shap_values, list):
        # Class 1 (Dropout) SHAP values
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    shap_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Mean |SHAP Value|": mean_abs_shap}
    ).sort_values(by="Mean |SHAP Value|", ascending=False)

    print("\n" + "=" * 70)
    print("   TOP 8 FEATURES BY MEAN ABSOLUTE SHAP VALUE (GRADIENT BOOSTING)   ")
    print("=" * 70)
    top_8_shap = shap_importance_df.head(8)
    for rank, (_, row) in enumerate(top_8_shap.iterrows(), 1):
        print(f"   {rank}. {row['Feature']:<45} : {row['Mean |SHAP Value|']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
