import os
import pandas as pd
import joblib
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(path):
    return pd.read_csv(path)

def main(data_path):
    df = load_data(data_path)
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print("Classification Report:\n", report)

    # Buat folder output lokal
    os.makedirs("outputs", exist_ok=True)
    local_model_path = "outputs/best_model.pkl"

    # Simpan model
    joblib.dump(model, local_model_path)

    # Logging lokal MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact("outputs/best_model.pkl")  # RELATIF ✅
        mlflow.sklearn.log_model(model, artifact_path="model")  # model_path internal mlflow

    print("✅ Model berhasil disimpan dan dilogging ke MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
