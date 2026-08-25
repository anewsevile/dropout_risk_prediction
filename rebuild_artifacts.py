import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

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

gradient_boosting_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
gradient_boosting_model.fit(X, y)

artifacts = {
    'model': gradient_boosting_model,
    'scaler': scaler,
    'ordinal_encoder': ordinal_encoder,
    'dept_encoder': dept_encoder,
    'category_encoder': category_encoder,
    'target_encoder': target_encoder
}

with open("dropout_prediction_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("Rebuilt with student_category feature.")