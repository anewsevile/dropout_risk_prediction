import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

# 1. Load dataset
df = pd.read_csv("student_dropout_risk_dataset.csv")

# 2. Setup Encoders & Scaler
dept_encoder = LabelEncoder()
target_encoder = LabelEncoder()
ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
scaler = StandardScaler()

# 3. Fit Encoders
df['dept_encoded'] = dept_encoder.fit_transform(df['department'])
df['target_encoded'] = target_encoder.fit_transform(df['dropout_risk_label'])
df['socio_encoded'] = ordinal_encoder.fit_transform(df[['socio_economic_proxy']]).ravel()

num_cols = [
    'year', 'attendance_percent', 'cgpa', 'internal_marks_avg',
    'assignment_submission_rate', 'backlogs', 'lms_engagement_score'
]

scaler.fit(df[num_cols])

X_num = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols)
X = X_num.copy()
X['department'] = df['dept_encoded']
X['socio_economic_proxy'] = df['socio_encoded']

feature_order = [
    'department', 'year', 'attendance_percent', 'cgpa',
    'internal_marks_avg', 'assignment_submission_rate',
    'backlogs', 'lms_engagement_score', 'socio_economic_proxy'
]
X = X[feature_order]
y = df['target_encoded']

# 4. Hardcode GradientBoostingClassifier selection
gradient_boosting_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
gradient_boosting_model.fit(X, y)

# 5. Package structure
artifacts = {
    'model': gradient_boosting_model,
    'scaler': scaler,
    'ordinal_encoder': ordinal_encoder,
    'dept_encoder': dept_encoder,
    'target_encoder': target_encoder
}

# 6. Save pkl file
with open("dropout_prediction_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

# 7. Reload and verify type
with open("dropout_prediction_artifacts.pkl", "rb") as f:
    loaded = pickle.load(f)

print("CONFIRMATION:")
print(type(loaded['model']))