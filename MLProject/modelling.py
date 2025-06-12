import os
import pandas as pd
import joblib
import argparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
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

    # Simpan model ke lokal (optional, untuk keperluan manual)
    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", "best_model.pkl")
    joblib.dump(model, model_path)

    # MLflow logging setup
    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name="Kriteria_3_Model"):
        # Log metric dan file artefak lokal (opsional)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(model_path)

        # Log model untuk kebutuhan Docker build
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, model.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # Penting untuk build-docker!
            input_example=input_example,
            signature=signature
        )

    print("✅ Model berhasil dilogging ke MLflow dan siap dibuild ke Docker.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
