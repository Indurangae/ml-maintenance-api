import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

#  Load enhanced dataset
df = pd.read_csv("enhanced_training_data.csv")  

#  Features and target labels
X = df[["tire_condition", "ac_condition", "hybrid_battery_condition", "engine_condition"]]

#   Train a separate model for each component
models = {}
for label in ["tire_maintenance", "ac_maintenance", "battery_maintenance", "engine_maintenance"]:
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    models[label] = clf
    print(f"\n📊 {label} model performance:")
    print(classification_report(y_test, clf.predict(X_test)))

#   Save the models
joblib.dump(models, "maintenance_models.pkl")
print("✅ All models trained and saved as maintenance_models.pkl")
