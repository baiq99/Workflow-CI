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

    # Simpan model ke folder outputs
    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", "best_model.pkl")
    joblib.dump(model, model_path)

    # Setup MLflow lokal
    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name="Kriteria_3_Model"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(model_path)

        # Tambahkan signature dan input_example agar build-docker tidak error
        input_example = X_test.iloc[:5]
        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

    print("✅ Model berhasil disimpan dan dilogging ke MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
