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

    # === Simpan model lokal ===
    os.makedirs("outputs", exist_ok=True)
    local_model_path = os.path.join("outputs", "best_model.pkl")
    joblib.dump(model, local_model_path)

    # === Logging ke MLflow DagsHub ===
    mlflow.set_tracking_uri("https://dagshub.com/baiq99/ml_flow.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

    mlflow.set_experiment("Eksperimen_Modeling_Advance")

    with mlflow.start_run(run_name="Kriteria_3_Model"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(local_model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("✅ Model berhasil dilogging ke DagsHub MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
