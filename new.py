import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

# ---------- 1. Load the data ----------
df = pd.read_csv("diabetes.csv")
print("Rows:", len(df))

# ---------- 2. Clean the data ----------
# In these columns, a 0 means "not measured". Replace it with the median value.
cols_with_fake_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_fake_zeros:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# ---------- 3. Separate inputs (X) and answer (y) ----------
X = df.drop(columns="Outcome")
y = df["Outcome"]

# ---------- 4. Split into training (80%) and test (20%) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 5. Scale the numbers ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 6. Train two models ----------
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_scaled, y_train)

forest_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
forest_model.fit(X_train_scaled, y_train)

# ---------- 7. Test both models ----------
for name, model in [("Logistic Regression", log_model), ("Random Forest", forest_model)]:
    predictions = model.predict(X_test_scaled)
    chances = model.predict_proba(X_test_scaled)[:, 1]
    print("\n", name)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Recall  :", round(recall_score(y_test, predictions), 3))
    print("ROC-AUC :", round(roc_auc_score(y_test, chances), 3))
    print(confusion_matrix(y_test, predictions))

# ---------- 8. Save the model and scaler for the app ----------
joblib.dump(log_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nSaved best_model.pkl and scaler.pkl")

# ---------- 9. Try one new person ----------
new_person = pd.DataFrame([{
    "Pregnancies": 2, "Glucose": 150, "BloodPressure": 80, "SkinThickness": 25,
    "Insulin": 120, "BMI": 32, "DiabetesPedigreeFunction": 0.6, "Age": 45,
}])
chance = log_model.predict_proba(scaler.transform(new_person))[0][1]
print("Chance of diabetes:", round(chance * 100, 1), "%")