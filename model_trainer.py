import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

np.random.seed(42)

n_samples = 2000
n_features = 10

normal_data = []
for _ in range(1000):
    beat = [0.1, 0.2, 0.5, 0.8, 1.0, 0.9, 0.4, 0.2, 0.1, 0.05]
    noise = np.random.normal(0, 0.15, n_features)
    normal_data.append(beat + noise)

abnormal_data = []
for _ in range(1000):
    beat = [0.1, 0.6, 0.3, 1.2, 0.7, 1.5, 0.2, 0.4, 0.8, 0.1]
    noise = np.random.normal(0, 0.25, n_features)
    abnormal_data.append(beat + noise)

X = np.vstack([normal_data, abnormal_data])
y = np.array([0]*1000 + [1]*1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'ecg_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"Model trained! Accuracy: {model.score(X_test_scaled, y_test):.2f}")
print("Files saved: ecg_model.pkl and scaler.pkl")