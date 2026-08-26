import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("student_dropout_risk_dataset.csv")

dept_encoder = LabelEncoder()
category_encoder = LabelEncoder()
target_encoder = LabelEncoder()
ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
scaler = StandardScaler()

df['dept_encoded'] = dept_encoder.fit_transform(df['department'])
df['category_encoded'] = category_encoder.fit_transform(df['student_category'])
df['target_encoded'] = target_encoder.fit_transform(df['dropout_risk_label'])
df['socio_encoded'] = ordinal_encoder.fit_transform(df[['socio_economic_proxy']]).ravel()

num_cols = ['year', 'attendance_percent', 'cgpa', 'internal_marks_avg',
            'assignment_submission_rate', 'backlogs', 'lms_engagement_score']
scaler.fit(df[num_cols])

X = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols)
X['department'] = df['dept_encoded']
X['socio_economic_proxy'] = df['socio_encoded']
X['student_category'] = df['category_encoded']

feature_order = ['department', 'year', 'attendance_percent', 'cgpa',
                  'internal_marks_avg', 'assignment_submission_rate',
                  'backlogs', 'lms_engagement_score', 'socio_economic_proxy',
                  'student_category']
X = X[feature_order]
y = df['target_encoded']

# Honest held-out evaluation (20% never seen during training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
eval_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
eval_model.fit(X_train, y_train)
y_pred_test = eval_model.predict(X_test)

eval_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_test),
    'precision': precision_score(y_test, y_pred_test, average='macro'),
    'recall': recall_score(y_test, y_pred_test, average='macro'),
    'confusion_matrix': confusion_matrix(
        y_test, y_pred_test,
        labels=[target_encoder.transform(['High'])[0],
                target_encoder.transform(['Medium'])[0],
                target_encoder.transform(['Low'])[0]]
    ),
    'cm_labels': ['High', 'Medium', 'Low']
}

# Final production model — trained on ALL data (for live predictions)
gradient_boosting_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
gradient_boosting_model.fit(X, y)

artifacts = {
    'model': gradient_boosting_model,
    'scaler': scaler,
    'ordinal_encoder': ordinal_encoder,
    'dept_encoder': dept_encoder,
    'category_encoder': category_encoder,
    'target_encoder': target_encoder,
    'eval_metrics': eval_metrics
}

with open("dropout_prediction_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("Rebuilt with student_category feature + honest eval metrics.")